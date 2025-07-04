# utils/validation.py
class ScoreValidator:
    @staticmethod
    def validate_score(score: float, name: str = "score") -> float:
        """Validate and clamp score to 0-1 range"""
        if not isinstance(score, (int, float)):
            raise ValueError(f"{name} must be numeric")
        return max(0.0, min(1.0, float(score)))
    
    @staticmethod
    def validate_percentage(value: float, name: str = "percentage") -> float:
        """Validate percentage (0-100)"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        return max(0.0, min(100.0, float(value)))