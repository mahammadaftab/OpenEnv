"""
Task Graders for OpenEnv

Implements agent graders for three difficulty levels (easy, medium, hard)
with scoring from 0.0 to 1.0 based on multiple criteria.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class GradingCriteria:
    """Individual grading criterion."""
    name: str
    weight: float
    score: float = 0.0


class TaskGrader:
    """
    Base class for task graders.
    
    Evaluates agent performance across multiple criteria
    and produces a normalized score between 0.0 and 1.0.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize grader with configuration.
        
        Args:
            config: Grading configuration from openenv.yaml
        """
        self.config = config
        self.criteria = []
        self.episode_data = {
            'steps': 0,
            'target_reached': False,
            'collisions': 0,
            'distance_traveled': 0.0,
            'energy_consumed': 0.0,
            'waypoints_passed': 0,
            'final_distance_to_target': float('inf'),
            'time_to_complete': 0,
            'max_wind_deviation': 0.0,
        }
        
        # Initialize criteria from config
        self._initialize_criteria()
    
    def _initialize_criteria(self) -> None:
        """Initialize grading criteria from configuration."""
        for criterion_config in self.config.get('criteria', []):
            criterion = GradingCriteria(
                name=criterion_config['name'],
                weight=criterion_config['weight']
            )
            self.criteria.append(criterion)
    
    def reset(self) -> None:
        """Reset episode data for new evaluation."""
        self.episode_data = {
            'steps': 0,
            'target_reached': False,
            'collisions': 0,
            'distance_traveled': 0.0,
            'energy_consumed': 0.0,
            'waypoints_passed': 0,
            'final_distance_to_target': float('inf'),
            'time_to_complete': 0,
            'max_wind_deviation': 0.0,
        }
        
        # Reset criterion scores
        for criterion in self.criteria:
            criterion.score = 0.0
    
    def update(self, **kwargs) -> None:
        """
        Update episode data with new information.
        
        Args:
            **kwargs: Episode metrics to update
        """
        for key, value in kwargs.items():
            if key in self.episode_data:
                # Handle special cases
                if key == 'collisions' and value > self.episode_data[key]:
                    self.episode_data[key] = value
                elif key == 'distance_traveled':
                    self.episode_data[key] += value
                else:
                    self.episode_data[key] = value
    
    def compute_scores(self) -> Dict[str, float]:
        """
        Compute individual criterion scores.
        
        Returns:
            Dictionary mapping criterion names to scores
        """
        raise NotImplementedError("Subclasses must implement compute_scores")
    
    def get_final_score(self) -> float:
        """
        Calculate weighted final score.
        
        Returns:
            Normalized score between 0.0 and 1.0
        """
        # First compute individual scores
        scores = self.compute_scores()
        
        # Calculate weighted average
        total_weight = sum(c.weight for c in self.criteria)
        weighted_sum = sum(c.score * c.weight for c in self.criteria)
        
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.0
        
        # Ensure score is in [0, 1] range
        final_score = np.clip(final_score, 0.0, 1.0)
        
        return final_score
    
    def get_grade_report(self) -> Dict[str, Any]:
        """
        Generate detailed grade report.
        
        Returns:
            Dictionary with scores, metadata, and feedback
        """
        scores = self.compute_scores()
        final_score = self.get_final_score()
        
        report = {
            'final_score': final_score,
            'success_threshold': self.config.get('success_threshold', 0.7),
            'passed': final_score >= self.config.get('success_threshold', 0.7),
            'criteria_scores': {c.name: c.score for c in self.criteria},
            'episode_data': self.episode_data.copy(),
            'feedback': self._generate_feedback(scores),
        }
        
        return report
    
    def _generate_feedback(self, scores: Dict[str, float]) -> str:
        """Generate human-readable feedback based on scores."""
        feedback_parts = []
        
        for criterion in self.criteria:
            score = scores.get(criterion.name, 0.0)
            if score < 0.5:
                feedback_parts.append(f"Needs improvement in {criterion.name}")
            elif score < 0.8:
                feedback_parts.append(f"Good performance in {criterion.name}")
            else:
                feedback_parts.append(f"Excellent {criterion.name}")
        
        return "; ".join(feedback_parts)


class EasyGrader(TaskGrader):
    """
    Grader for easy task: Basic Navigation.
    
    Criteria:
    - Reached target (60%)
    - Time efficiency (20%)
    - Energy efficiency (20%)
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """Compute scores for easy task criteria."""
        scores = {}
        
        # Target reached (60%)
        target_criterion = next((c for c in self.criteria if c.name == 'reached_target'), None)
        if target_criterion:
            if self.episode_data['target_reached']:
                target_criterion.score = 1.0
            else:
                # Partial credit based on proximity
                max_dist = 80.0  # boundary limit
                actual_dist = self.episode_data['final_distance_to_target']
                target_criterion.score = max(0.0, 1.0 - (actual_dist / max_dist))
        scores['reached_target'] = target_criterion.score if target_criterion else 0.0
        
        # Time efficiency (20%)
        time_criterion = next((c for c in self.criteria if c.name == 'time_efficiency'), None)
        if time_criterion:
            max_steps = 300
            actual_steps = self.episode_data['steps']
            if self.episode_data['target_reached']:
                # Faster is better
                time_criterion.score = max(0.0, 1.0 - (actual_steps / max_steps))
            else:
                time_criterion.score = 0.3  # Minimum credit for trying
        scores['time_efficiency'] = time_criterion.score if time_criterion else 0.0
        
        # Energy efficiency (20%)
        energy_criterion = next((c for c in self.criteria if c.name == 'energy_efficiency'), None)
        if energy_criterion:
            # Based on distance traveled vs optimal path
            optimal_distance = self.episode_data.get('optimal_distance', 50.0)
            actual_distance = self.episode_data['distance_traveled']
            if actual_distance > 0:
                energy_criterion.score = min(1.0, optimal_distance / actual_distance)
            else:
                energy_criterion.score = 0.0
        scores['energy_efficiency'] = energy_criterion.score if energy_criterion else 0.0
        
        return scores


class MediumGrader(TaskGrader):
    """
    Grader for medium task: Obstacle Avoidance.
    
    Criteria:
    - Reached target (50%)
    - Collision avoidance (25%)
    - Time efficiency (15%)
    - Energy efficiency (10%)
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """Compute scores for medium task criteria."""
        scores = {}
        
        # Target reached (50%)
        target_criterion = next((c for c in self.criteria if c.name == 'reached_target'), None)
        if target_criterion:
            if self.episode_data['target_reached']:
                target_criterion.score = 1.0
            else:
                max_dist = 60.0
                actual_dist = self.episode_data['final_distance_to_target']
                target_criterion.score = max(0.0, 1.0 - (actual_dist / max_dist))
        scores['reached_target'] = target_criterion.score if target_criterion else 0.0
        
        # Collision avoidance (25%)
        collision_criterion = next((c for c in self.criteria if c.name == 'collision_avoidance'), None)
        if collision_criterion:
            max_collisions = 5
            actual_collisions = self.episode_data['collisions']
            if actual_collisions == 0:
                collision_criterion.score = 1.0
            else:
                collision_criterion.score = max(0.0, 1.0 - (actual_collisions / max_collisions))
        scores['collision_avoidance'] = collision_criterion.score if collision_criterion else 0.0
        
        # Time efficiency (15%)
        time_criterion = next((c for c in self.criteria if c.name == 'time_efficiency'), None)
        if time_criterion:
            max_steps = 500
            actual_steps = self.episode_data['steps']
            if self.episode_data['target_reached']:
                time_criterion.score = max(0.0, 1.0 - (actual_steps / max_steps))
            else:
                time_criterion.score = 0.3
        scores['time_efficiency'] = time_criterion.score if time_criterion else 0.0
        
        # Energy efficiency (10%)
        energy_criterion = next((c for c in self.criteria if c.name == 'energy_efficiency'), None)
        if energy_criterion:
            optimal_distance = self.episode_data.get('optimal_distance', 40.0)
            actual_distance = self.episode_data['distance_traveled']
            if actual_distance > 0:
                energy_criterion.score = min(1.0, optimal_distance / actual_distance)
            else:
                energy_criterion.score = 0.0
        scores['energy_efficiency'] = energy_criterion.score if energy_criterion else 0.0
        
        return scores


class HardGrader(TaskGrader):
    """
    Grader for hard task: Dynamic Environment.
    
    Criteria:
    - Reached target (45%)
    - Collision avoidance (25%)
    - Wind compensation (15%)
    - Time efficiency (10%)
    - Energy efficiency (5%)
    """
    
    def compute_scores(self) -> Dict[str, float]:
        """Compute scores for hard task criteria."""
        scores = {}
        
        # Target reached (45%)
        target_criterion = next((c for c in self.criteria if c.name == 'reached_target'), None)
        if target_criterion:
            if self.episode_data['target_reached']:
                target_criterion.score = 1.0
            else:
                max_dist = 50.0
                actual_dist = self.episode_data['final_distance_to_target']
                target_criterion.score = max(0.0, 1.0 - (actual_dist / max_dist))
        scores['reached_target'] = target_criterion.score if target_criterion else 0.0
        
        # Collision avoidance (25%)
        collision_criterion = next((c for c in self.criteria if c.name == 'collision_avoidance'), None)
        if collision_criterion:
            max_collisions = 10
            actual_collisions = self.episode_data['collisions']
            if actual_collisions == 0:
                collision_criterion.score = 1.0
            else:
                collision_criterion.score = max(0.0, 1.0 - (actual_collisions / max_collisions))
        scores['collision_avoidance'] = collision_criterion.score if collision_criterion else 0.0
        
        # Wind compensation (15%)
        wind_criterion = next((c for c in self.criteria if c.name == 'wind_compensation'), None)
        if wind_criterion:
            # Score based on how well agent maintained course despite wind
            max_deviation = 20.0
            actual_deviation = self.episode_data['max_wind_deviation']
            wind_criterion.score = max(0.0, 1.0 - (actual_deviation / max_deviation))
        scores['wind_compensation'] = wind_criterion.score if wind_criterion else 0.0
        
        # Time efficiency (10%)
        time_criterion = next((c for c in self.criteria if c.name == 'time_efficiency'), None)
        if time_criterion:
            max_steps = 700
            actual_steps = self.episode_data['steps']
            if self.episode_data['target_reached']:
                time_criterion.score = max(0.0, 1.0 - (actual_steps / max_steps))
            else:
                time_criterion.score = 0.3
        scores['time_efficiency'] = time_criterion.score if time_criterion else 0.0
        
        # Energy efficiency (5%)
        energy_criterion = next((c for c in self.criteria if c.name == 'energy_efficiency'), None)
        if energy_criterion:
            optimal_distance = self.episode_data.get('optimal_distance', 35.0)
            actual_distance = self.episode_data['distance_traveled']
            if actual_distance > 0:
                energy_criterion.score = min(1.0, optimal_distance / actual_distance)
            else:
                energy_criterion.score = 0.0
        scores['energy_efficiency'] = energy_criterion.score if energy_criterion else 0.0
        
        return scores


def create_grader(task_level: str, config: Dict[str, Any]) -> TaskGrader:
    """
    Factory function to create appropriate grader for task level.
    
    Args:
        task_level: Difficulty level ('easy', 'medium', 'hard')
        config: Grading configuration
        
    Returns:
        Appropriate TaskGrader instance
    """
    graders = {
        'easy': EasyGrader,
        'medium': MediumGrader,
        'hard': HardGrader,
    }
    
    if task_level not in graders:
        raise ValueError(f"Unknown task level: {task_level}. Must be one of {list(graders.keys())}")
    
    return graders[task_level](config)
