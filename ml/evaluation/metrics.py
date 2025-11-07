"""
Evaluation metrics for music structure recognition.

Implements boundary detection metrics (F@0.5s, F@3s) and labeling accuracy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class BoundaryMetrics:
    """Boundary detection metrics."""

    precision: float
    recall: float
    f1: float
    hit_rate: float
    num_predicted: int
    num_ground_truth: int
    num_hits: int


@dataclass
class LabelMetrics:
    """Label classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    class_report: Dict[str, Dict[str, float]]


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    boundary_metrics_05: BoundaryMetrics
    boundary_metrics_30: BoundaryMetrics
    label_metrics: LabelMetrics
    track_results: List[Dict[str, any]]


class BoundaryEvaluator:
    """Evaluates boundary detection performance."""

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance

    def evaluate(
        self, predicted_boundaries: List[float], ground_truth_boundaries: List[float]
    ) -> BoundaryMetrics:
        """
        Evaluate boundary detection performance.

        Args:
            predicted_boundaries: List of predicted boundary times
            ground_truth_boundaries: List of ground truth boundary times

        Returns:
            Boundary detection metrics
        """
        if not predicted_boundaries and not ground_truth_boundaries:
            return BoundaryMetrics(1.0, 1.0, 1.0, 1.0, 0, 0, 0)

        if not predicted_boundaries:
            return BoundaryMetrics(
                0.0, 0.0, 0.0, 0.0, 0, len(ground_truth_boundaries), 0
            )

        if not ground_truth_boundaries:
            return BoundaryMetrics(0.0, 0.0, 0.0, 0.0, len(predicted_boundaries), 0, 0)

        # Find hits within tolerance
        hits = self._find_hits(predicted_boundaries, ground_truth_boundaries)

        # Calculate metrics
        precision = hits / len(predicted_boundaries) if predicted_boundaries else 0.0
        recall = hits / len(ground_truth_boundaries) if ground_truth_boundaries else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        hit_rate = (
            hits / len(ground_truth_boundaries) if ground_truth_boundaries else 0.0
        )

        return BoundaryMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            hit_rate=hit_rate,
            num_predicted=len(predicted_boundaries),
            num_ground_truth=len(ground_truth_boundaries),
            num_hits=hits,
        )

    def _find_hits(self, predicted: List[float], ground_truth: List[float]) -> int:
        """Find number of hits within tolerance."""
        hits = 0
        used_gt = set()

        for pred_boundary in predicted:
            for i, gt_boundary in enumerate(ground_truth):
                if i in used_gt:
                    continue

                if abs(pred_boundary - gt_boundary) <= self.tolerance:
                    hits += 1
                    used_gt.add(i)
                    break

        return hits


class LabelEvaluator:
    """Evaluates label classification performance."""

    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.label_to_idx = {label: i for i, label in enumerate(label_names)}

    def evaluate(
        self, predicted_labels: List[str], ground_truth_labels: List[str]
    ) -> LabelMetrics:
        """
        Evaluate label classification performance.

        Args:
            predicted_labels: List of predicted labels
            ground_truth_labels: List of ground truth labels

        Returns:
            Label classification metrics
        """
        if not predicted_labels or not ground_truth_labels:
            return LabelMetrics(0.0, 0.0, 0.0, 0.0, np.array([]), {})

        # Convert to indices
        pred_indices = [
            self.label_to_idx.get(label, 7) for label in predicted_labels
        ]  # Default to OTHER
        gt_indices = [self.label_to_idx.get(label, 7) for label in ground_truth_labels]

        # Calculate accuracy
        accuracy = accuracy_score(gt_indices, pred_indices)

        # Calculate precision, recall, F1
        precision = np.mean(
            [
                self._precision_score(gt_indices, pred_indices, i)
                for i in range(len(self.label_names))
            ]
        )
        recall = np.mean(
            [
                self._recall_score(gt_indices, pred_indices, i)
                for i in range(len(self.label_names))
            ]
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Confusion matrix
        cm = confusion_matrix(
            gt_indices, pred_indices, labels=range(len(self.label_names))
        )

        # Classification report
        class_report = classification_report(
            gt_indices,
            pred_indices,
            labels=range(len(self.label_names)),
            target_names=self.label_names,
            output_dict=True,
            zero_division=0,
        )

        return LabelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            class_report=class_report,
        )

    def _precision_score(
        self, y_true: List[int], y_pred: List[int], class_idx: int
    ) -> float:
        """Calculate precision for a specific class."""
        tp = sum(
            1
            for i in range(len(y_pred))
            if y_pred[i] == class_idx and y_true[i] == class_idx
        )
        fp = sum(
            1
            for i in range(len(y_pred))
            if y_pred[i] == class_idx and y_true[i] != class_idx
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _recall_score(
        self, y_true: List[int], y_pred: List[int], class_idx: int
    ) -> float:
        """Calculate recall for a specific class."""
        tp = sum(
            1
            for i in range(len(y_pred))
            if y_pred[i] == class_idx and y_true[i] == class_idx
        )
        fn = sum(
            1
            for i in range(len(y_true))
            if y_true[i] == class_idx and y_pred[i] != class_idx
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class SegmentEvaluator:
    """Evaluates segment-level performance."""

    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.boundary_evaluator_05 = BoundaryEvaluator(tolerance=0.5)
        self.boundary_evaluator_30 = BoundaryEvaluator(tolerance=3.0)
        self.label_evaluator = LabelEvaluator(label_names)

    def evaluate_track(
        self,
        predicted_boundaries: List[float],
        predicted_labels: List[str],
        ground_truth_boundaries: List[float],
        ground_truth_labels: List[str],
    ) -> Dict[str, any]:
        """
        Evaluate a single track.

        Args:
            predicted_boundaries: Predicted boundary times
            predicted_labels: Predicted labels
            ground_truth_boundaries: Ground truth boundary times
            ground_truth_labels: Ground truth labels

        Returns:
            Dictionary with track-level metrics
        """
        # Boundary evaluation
        boundary_metrics_05 = self.boundary_evaluator_05.evaluate(
            predicted_boundaries, ground_truth_boundaries
        )
        boundary_metrics_30 = self.boundary_evaluator_30.evaluate(
            predicted_boundaries, ground_truth_boundaries
        )

        # Label evaluation
        label_metrics = self.label_evaluator.evaluate(
            predicted_labels, ground_truth_labels
        )

        return {
            "boundary_metrics_05": boundary_metrics_05,
            "boundary_metrics_30": boundary_metrics_30,
            "label_metrics": label_metrics,
        }

    def evaluate_dataset(
        self, track_results: List[Dict[str, any]]
    ) -> EvaluationResults:
        """
        Evaluate entire dataset.

        Args:
            track_results: List of track-level results

        Returns:
            Aggregated evaluation results
        """
        # Aggregate boundary metrics
        boundary_metrics_05 = self._aggregate_boundary_metrics(
            [r["boundary_metrics_05"] for r in track_results]
        )
        boundary_metrics_30 = self._aggregate_boundary_metrics(
            [r["boundary_metrics_30"] for r in track_results]
        )

        # Aggregate label metrics
        label_metrics = self._aggregate_label_metrics(
            [r["label_metrics"] for r in track_results]
        )

        return EvaluationResults(
            boundary_metrics_05=boundary_metrics_05,
            boundary_metrics_30=boundary_metrics_30,
            label_metrics=label_metrics,
            track_results=track_results,
        )

    def _aggregate_boundary_metrics(
        self, metrics_list: List[BoundaryMetrics]
    ) -> BoundaryMetrics:
        """Aggregate boundary metrics across tracks."""
        total_predicted = sum(m.num_predicted for m in metrics_list)
        total_ground_truth = sum(m.num_ground_truth for m in metrics_list)
        total_hits = sum(m.num_hits for m in metrics_list)

        precision = total_hits / total_predicted if total_predicted > 0 else 0.0
        recall = total_hits / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        hit_rate = total_hits / total_ground_truth if total_ground_truth > 0 else 0.0

        return BoundaryMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            hit_rate=hit_rate,
            num_predicted=total_predicted,
            num_ground_truth=total_ground_truth,
            num_hits=total_hits,
        )

    def _aggregate_label_metrics(
        self, metrics_list: List[LabelMetrics]
    ) -> LabelMetrics:
        """Aggregate label metrics across tracks."""
        if not metrics_list:
            return LabelMetrics(0.0, 0.0, 0.0, 0.0, np.array([]), {})

        # Average metrics
        accuracy = np.mean([m.accuracy for m in metrics_list])
        precision = np.mean([m.precision for m in metrics_list])
        recall = np.mean([m.recall for m in metrics_list])
        f1 = np.mean([m.f1 for m in metrics_list])

        # Sum confusion matrices
        total_cm = sum(m.confusion_matrix for m in metrics_list)

        # Average classification report
        class_report = {}
        for class_name in self.label_names:
            class_report[class_name] = {
                "precision": np.mean(
                    [
                        m.class_report.get(class_name, {}).get("precision", 0.0)
                        for m in metrics_list
                    ]
                ),
                "recall": np.mean(
                    [
                        m.class_report.get(class_name, {}).get("recall", 0.0)
                        for m in metrics_list
                    ]
                ),
                "f1-score": np.mean(
                    [
                        m.class_report.get(class_name, {}).get("f1-score", 0.0)
                        for m in metrics_list
                    ]
                ),
            }

        return LabelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=total_cm,
            class_report=class_report,
        )


def print_evaluation_results(results: EvaluationResults):
    """Print evaluation results in a readable format."""
    print("=" * 60)
    print("MUSIC STRUCTURE RECOGNITION EVALUATION RESULTS")
    print("=" * 60)

    # Boundary detection results
    print("\nBOUNDARY DETECTION (F@0.5s):")
    print(f"  Precision: {results.boundary_metrics_05.precision:.3f}")
    print(f"  Recall:    {results.boundary_metrics_05.recall:.3f}")
    print(f"  F1:        {results.boundary_metrics_05.f1:.3f}")
    print(f"  Hit Rate:  {results.boundary_metrics_05.hit_rate:.3f}")

    print("\nBOUNDARY DETECTION (F@3.0s):")
    print(f"  Precision: {results.boundary_metrics_30.precision:.3f}")
    print(f"  Recall:    {results.boundary_metrics_30.recall:.3f}")
    print(f"  F1:        {results.boundary_metrics_30.f1:.3f}")
    print(f"  Hit Rate:  {results.boundary_metrics_30.hit_rate:.3f}")

    # Label classification results
    print("\nLABEL CLASSIFICATION:")
    print(f"  Accuracy:  {results.label_metrics.accuracy:.3f}")
    print(f"  Precision: {results.label_metrics.precision:.3f}")
    print(f"  Recall:    {results.label_metrics.recall:.3f}")
    print(f"  F1:        {results.label_metrics.f1:.3f}")

    # Per-class results
    print("\nPER-CLASS RESULTS:")
    for class_name, metrics in results.label_metrics.class_report.items():
        print(
            f"  {class_name:10}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}"
        )

    print("=" * 60)


def main():
    """Test the evaluation metrics."""
    # Test data
    predicted_boundaries = [0.0, 12.5, 25.0, 37.5, 50.0]
    predicted_labels = ["INTRO", "VERSE", "CHORUS", "VERSE", "OUTRO"]

    ground_truth_boundaries = [0.0, 12.0, 25.5, 38.0, 50.0]
    ground_truth_labels = ["INTRO", "VERSE", "CHORUS", "VERSE", "OUTRO"]

    label_names = [
        "INTRO",
        "VERSE",
        "PRE",
        "CHORUS",
        "BRIDGE",
        "SOLO",
        "OUTRO",
        "OTHER",
    ]

    # Evaluate
    evaluator = SegmentEvaluator(label_names)
    result = evaluator.evaluate_track(
        predicted_boundaries,
        predicted_labels,
        ground_truth_boundaries,
        ground_truth_labels,
    )

    # Print results
    print("Single track evaluation:")
    print(f"Boundary F@0.5s: {result['boundary_metrics_05'].f1:.3f}")
    print(f"Boundary F@3.0s: {result['boundary_metrics_30'].f1:.3f}")
    print(f"Label accuracy: {result['label_metrics'].accuracy:.3f}")


if __name__ == "__main__":
    main()
