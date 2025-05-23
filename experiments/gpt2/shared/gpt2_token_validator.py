"""
GPT-2 Token Validation System

Comprehensive validation system for identifying single BPE tokens in GPT-2's vocabulary.
Essential for semantic subtype experiments to ensure we're analyzing complete semantic
concepts rather than fragmented subword pieces.

Key Features:
- Single-token validation with detailed token breakdowns and error handling
- Word curation to select exactly N single-token words from candidate pools
- Fallback strategies for insufficient primary candidates
- Batch validation with comparative statistics across word categories
- Distribution analysis of token counts and complexity patterns
- Progress reporting for long operations

Classes:
- ValidationResult: Detailed results for single word validation
- CurationResult: Results and statistics for word list curation
- GPT2TokenValidator: Main validation, curation, and analysis engine

Integration:
- Designed for GPT-2 Semantic Subtypes experiment (TODO-1.1)
- Compatible with existing APA framework infrastructure
- Supports downstream clustering and probing classifier pipelines
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single word against GPT-2 tokenizer."""
    word: str
    is_single_token: bool
    token_count: int
    token_breakdown: List[str]  # Human-readable tokens
    token_ids: List[int]        # Numeric token IDs
    error_message: Optional[str] = None


@dataclass
class CurationResult:
    """Result of curating a wordlist to contain only single-token words."""
    valid_words: List[str]              # Words that are single tokens
    invalid_words: List[ValidationResult]  # Words that are multi-token or errored
    statistics: Dict[str, Any]          # Summary statistics
    target_count: int                   # Requested number of words
    achieved_count: int                 # Actual number of words found
    success: bool                       # Whether target was achieved


class GPT2TokenValidator:
    """
    Validates words are single BPE tokens for GPT-2 Small.
    
    Essential for semantic experiments to ensure we analyze complete
    concepts rather than subword pieces like "free" + "dom" for "freedom".
    """
    
    def __init__(self):
        """Initialize validator with GPT-2 Small tokenizer."""
        self.tokenizer = None
        self._setup_tokenizer()
    
    def _setup_tokenizer(self):
        """Setup GPT-2 tokenizer using existing patterns."""
        try:
            from transformers import GPT2Tokenizer
            
            logger.info("Loading GPT-2 tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            # Set padding token following existing pattern
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("GPT-2 tokenizer loaded successfully")
            
        except ImportError as e:
            error_msg = f"Could not import transformers library: {e}"
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load GPT-2 tokenizer: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def validate_word(self, word: str) -> ValidationResult:
        """
        Validate that a word is exactly one BPE token.
        
        Args:
            word: The word to validate
            
        Returns:
            ValidationResult with detailed breakdown
            
        Example:
            validate_word("cat") -> ValidationResult(
                word="cat", 
                is_single_token=True, 
                token_count=1,
                token_breakdown=["cat"], 
                token_ids=[5246]
            )
            
            validate_word("freedom") -> ValidationResult(
                word="freedom",
                is_single_token=False,
                token_count=2, 
                token_breakdown=["free", "dom"],
                token_ids=[5787, 3438]
            )
        """
        if not word or not isinstance(word, str):
            return ValidationResult(
                word=str(word),
                is_single_token=False,
                token_count=0,
                token_breakdown=[],
                token_ids=[],
                error_message="Invalid input: word must be non-empty string"
            )
        
        try:
            # Tokenize without special tokens (no <|endoftext|> etc.)
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            
            # Convert token IDs back to text for human readability
            token_breakdown = []
            for token_id in token_ids:
                try:
                    token_text = self.tokenizer.decode([token_id])
                    token_breakdown.append(token_text)
                except Exception:
                    # Fallback for problematic tokens
                    token_breakdown.append(f"<token_{token_id}>")
            
            # Determine if single token
            is_single = len(token_ids) == 1
            
            return ValidationResult(
                word=word,
                is_single_token=is_single,
                token_count=len(token_ids),
                token_breakdown=token_breakdown,
                token_ids=token_ids,
                error_message=None
            )
            
        except Exception as e:
            error_msg = f"Tokenization failed for '{word}': {str(e)}"
            logger.warning(error_msg)
            
            return ValidationResult(
                word=word,
                is_single_token=False,
                token_count=0,
                token_breakdown=[],
                token_ids=[],
                error_message=error_msg
            )
    
    def validate_words(self, words: List[str]) -> List[ValidationResult]:
        """
        Validate multiple words efficiently.
        
        Args:
            words: List of words to validate
            
        Returns:
            List of ValidationResult objects
        """
        if not words:
            return []
        
        results = []
        for word in words:
            result = self.validate_word(word)
            results.append(result)
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate summary statistics from validation results.
        
        Args:
            results: List of ValidationResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "total_words": 0,
                "single_token_count": 0,
                "multi_token_count": 0,
                "error_count": 0,
                "success_rate": 0.0
            }
        
        total_words = len(results)
        single_token_count = sum(1 for r in results if r.is_single_token)
        multi_token_count = sum(1 for r in results if not r.is_single_token and r.error_message is None)
        error_count = sum(1 for r in results if r.error_message is not None)
        
        success_rate = single_token_count / total_words if total_words > 0 else 0.0
        
        return {
            "total_words": total_words,
            "single_token_count": single_token_count,
            "multi_token_count": multi_token_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "success_percentage": success_rate * 100
        }
    
    def batch_validate(self, word_lists: Dict[str, List[str]], show_progress: bool = True) -> Dict[str, Dict]:
        """
        Validate multiple word lists and generate comparative statistics.
        
        Args:
            word_lists: Dictionary mapping category names to word lists
            show_progress: Whether to show progress information
            
        Returns:
            Dictionary with validation results and comparative statistics
            
        Example:
            results = validator.batch_validate({
                "nouns": ["cat", "dog", "smartphone"],
                "verbs": ["run", "understand", "think"]
            })
        """
        if show_progress:
            print(f"Batch validating {len(word_lists)} word categories...")
        
        category_results = {}
        all_validation_results = []
        
        for category, words in word_lists.items():
            if show_progress:
                print(f"  Validating {category}: {len(words)} words")
            
            validation_results = [self.validate_word(word) for word in words]
            category_stats = self.get_validation_summary(validation_results)
            
            category_results[category] = {
                "validation_results": validation_results,
                "statistics": category_stats,
                "word_count": len(words),
                "valid_words": [r.word for r in validation_results if r.is_single_token],
                "invalid_words": [r.word for r in validation_results if not r.is_single_token]
            }
            
            all_validation_results.extend(validation_results)
        
        overall_stats = self.get_validation_summary(all_validation_results)
        
        comparative_stats = {
            "category_comparison": {
                cat: {
                    "success_rate": results["statistics"]["success_rate"],
                    "word_count": results["word_count"],
                    "valid_count": results["statistics"]["single_token_count"]
                }
                for cat, results in category_results.items()
            },
            "best_category": max(category_results.keys(), 
                               key=lambda k: category_results[k]["statistics"]["success_rate"])
                              if category_results else None,
            "worst_category": min(category_results.keys(), 
                                key=lambda k: category_results[k]["statistics"]["success_rate"])
                               if category_results else None
        }
        
        return {
            "category_results": category_results,
            "overall_statistics": overall_stats,
            "comparative_statistics": comparative_stats,
            "summary": {
                "total_categories": len(word_lists),
                "total_words": sum(len(words) for words in word_lists.values()),
                "overall_success_rate": overall_stats["success_rate"]
            }
        }
    
    def generate_distribution_report(self, validation_results: List[ValidationResult]) -> Dict:
        """
        Generate detailed distribution analysis of validation results.
        
        Args:
            validation_results: List of ValidationResult objects
            
        Returns:
            Dictionary containing distribution statistics
        """
        if not validation_results:
            return {"error": "No validation results provided"}
        
        token_counts = [r.token_count for r in validation_results if r.token_count is not None]
        
        from collections import Counter
        token_distribution = Counter(token_counts)
        
        return {
            "token_count_distribution": dict(token_distribution),
            "most_common_token_count": token_distribution.most_common(1)[0] if token_distribution else None,
            "token_count_range": {
                "min": min(token_counts) if token_counts else None,
                "max": max(token_counts) if token_counts else None,
                "mean": sum(token_counts) / len(token_counts) if token_counts else None
            },
            "complexity_analysis": {
                "single_token_words": [r.word for r in validation_results if r.is_single_token],
                "complex_words": [
                    {"word": r.word, "token_count": r.token_count, "tokens": r.token_breakdown}
                    for r in validation_results 
                    if not r.is_single_token and r.token_breakdown
                ],
                "failed_words": [
                    {"word": r.word, "error": r.error_message}
                    for r in validation_results 
                    if r.error_message
                ]
            }
        }
    
    def curate_wordlist(self, candidate_words: List[str], target_count: int, 
                       show_progress: bool = True) -> CurationResult:
        """
        Curate a wordlist to contain exactly target_count single-token words.
        
        Args:
            candidate_words: List of candidate words to validate
            target_count: Desired number of single-token words
            show_progress: Whether to show progress for long operations
            
        Returns:
            CurationResult with selected words and statistics
            
        Example:
            candidates = ["cat", "dog", "smartphone", "house", "neighborhood"]
            result = validator.curate_wordlist(candidates, target_count=3)
            # result.valid_words = ["cat", "dog", "house"]
            # result.invalid_words contains ValidationResults for "smartphone", "neighborhood"
        """
        if not candidate_words:
            return CurationResult(
                valid_words=[],
                invalid_words=[],
                statistics={"error": "No candidate words provided"},
                target_count=target_count,
                achieved_count=0,
                success=False
            )
        
        if target_count <= 0:
            return CurationResult(
                valid_words=[],
                invalid_words=[],
                statistics={"error": "Target count must be positive"},
                target_count=target_count,
                achieved_count=0,
                success=False
            )
        
        valid_words = []
        invalid_words = []
        processed_count = 0
        
        logger.info(f"Curating wordlist: seeking {target_count} single-token words from {len(candidate_words)} candidates")
        
        for word in candidate_words:
            processed_count += 1
            
            # Show progress for long operations
            if show_progress and processed_count % 50 == 0:
                logger.info(f"Progress: processed {processed_count}/{len(candidate_words)}, found {len(valid_words)} valid words")
            
            # Validate word
            result = self.validate_word(word)
            
            if result.is_single_token:
                valid_words.append(word)
                
                # Stop when we have enough valid words
                if len(valid_words) >= target_count:
                    break
            else:
                invalid_words.append(result)
        
        # Generate statistics
        achieved_count = len(valid_words)
        success = achieved_count >= target_count
        
        statistics = {
            "candidates_processed": processed_count,
            "total_candidates": len(candidate_words),
            "valid_found": achieved_count,
            "invalid_found": len(invalid_words),
            "target_achieved": success,
            "efficiency": achieved_count / processed_count if processed_count > 0 else 0.0
        }
        
        if success:
            logger.info(f"Successfully curated {achieved_count} single-token words from {processed_count} candidates")
        else:
            logger.warning(f"Only found {achieved_count}/{target_count} single-token words from {len(candidate_words)} candidates")
        
        return CurationResult(
            valid_words=valid_words[:target_count],  # Ensure exact count
            invalid_words=invalid_words,
            statistics=statistics,
            target_count=target_count,
            achieved_count=achieved_count,
            success=success
        )
    
    def curate_with_fallback(self, primary_candidates: List[str], 
                           fallback_candidates: List[str], 
                           target_count: int) -> CurationResult:
        """
        Curate wordlist with fallback strategy for insufficient primary candidates.
        
        Args:
            primary_candidates: Preferred candidate words
            fallback_candidates: Backup candidates if primary insufficient
            target_count: Desired number of single-token words
            
        Returns:
            CurationResult with selected words and fallback usage statistics
        """
        logger.info(f"Curating with fallback: {len(primary_candidates)} primary + {len(fallback_candidates)} fallback candidates")
        
        # Try primary candidates first
        primary_result = self.curate_wordlist(primary_candidates, target_count, show_progress=False)
        
        if primary_result.success:
            # Primary candidates sufficient
            primary_result.statistics["fallback_used"] = False
            primary_result.statistics["fallback_count"] = 0
            return primary_result
        
        # Need fallback candidates
        remaining_needed = target_count - primary_result.achieved_count
        logger.info(f"Primary candidates yielded {primary_result.achieved_count}/{target_count}, seeking {remaining_needed} from fallback")
        
        # Curate from fallback candidates
        fallback_result = self.curate_wordlist(fallback_candidates, remaining_needed, show_progress=False)
        
        # Combine results
        combined_valid = primary_result.valid_words + fallback_result.valid_words
        combined_invalid = primary_result.invalid_words + fallback_result.invalid_words
        
        total_achieved = len(combined_valid)
        final_success = total_achieved >= target_count
        
        combined_statistics = {
            "primary_processed": primary_result.statistics["candidates_processed"],
            "fallback_processed": fallback_result.statistics["candidates_processed"], 
            "total_processed": primary_result.statistics["candidates_processed"] + fallback_result.statistics["candidates_processed"],
            "primary_valid": primary_result.achieved_count,
            "fallback_valid": fallback_result.achieved_count,
            "total_valid": total_achieved,
            "fallback_used": True,
            "fallback_count": fallback_result.achieved_count,
            "target_achieved": final_success
        }
        
        return CurationResult(
            valid_words=combined_valid[:target_count],  # Ensure exact count
            invalid_words=combined_invalid,
            statistics=combined_statistics,
            target_count=target_count,
            achieved_count=total_achieved,
            success=final_success
        )


def test_validator():
    """Test the validator with known examples."""
    validator = GPT2TokenValidator()
    
    # Test cases with expected outcomes
    test_cases = [
        ("cat", True),           # Common single token
        ("dog", True),           # Common single token  
        ("smartphone", False),   # "smart" + "phone"
        ("neighborhood", False), # "ne" + "igh" + "bor" + "hood"
        ("the", True),           # Common single token
        ("", False),             # Empty string (error case)
        ("hello", True),         # Common single token
    ]
    
    print("=== GPT-2 Token Validator Test ===")
    
    for word, expected_single in test_cases:
        result = validator.validate_word(word)
        status = "PASS" if result.is_single_token == expected_single else "FAIL"
        
        print(f"{status} '{word}' -> single_token={result.is_single_token}, tokens={result.token_breakdown}")
        
        if result.error_message:
            print(f"    Error: {result.error_message}")
    
    # Test summary statistics
    all_results = [validator.validate_word(word) for word, _ in test_cases]
    summary = validator.get_validation_summary(all_results)
    
    print(f"\n=== Summary ===")
    print(f"Total words: {summary['total_words']}")
    print(f"Single tokens: {summary['single_token_count']}")
    print(f"Multi tokens: {summary['multi_token_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Success rate: {summary['success_percentage']:.1f}%")


def test_curation():
    """Test the word curation system."""
    validator = GPT2TokenValidator()
    
    # Test case: mix of single and multi-token words
    candidates = [
        "cat", "dog", "house", "car", "tree",           # Single tokens (hopefully)
        "smartphone", "neighborhood", "understanding",   # Multi-token
        "book", "chair", "table", "phone", "water"      # More single tokens
    ]
    
    print("\n=== Word Curation Test ===")
    print(f"Candidates: {candidates}")
    
    # Test normal curation
    result = validator.curate_wordlist(candidates, target_count=5, show_progress=False)
    
    print(f"\nCuration Result:")
    print(f"Target: {result.target_count}")
    print(f"Achieved: {result.achieved_count}")
    print(f"Success: {result.success}")
    print(f"Valid words: {result.valid_words}")
    print(f"Invalid count: {len(result.invalid_words)}")
    print(f"Statistics: {result.statistics}")
    
    # Test fallback curation
    primary = ["smartphone", "neighborhood", "understanding"]  # Mostly multi-token
    fallback = ["cat", "dog", "house", "car", "tree"]         # Mostly single-token
    
    print(f"\n=== Fallback Curation Test ===")
    fallback_result = validator.curate_with_fallback(primary, fallback, target_count=3)
    
    print(f"Fallback Result:")
    print(f"Target: {fallback_result.target_count}")
    print(f"Achieved: {fallback_result.achieved_count}")
    print(f"Success: {fallback_result.success}")
    print(f"Valid words: {fallback_result.valid_words}")
    print(f"Fallback used: {fallback_result.statistics.get('fallback_used', False)}")
    
    # Test edge cases
    print(f"\n=== Edge Case Tests ===")
    
    # Empty candidates
    empty_result = validator.curate_wordlist([], target_count=5)
    print(f"Empty candidates: success={empty_result.success}")
    
    # Zero target
    zero_result = validator.curate_wordlist(candidates, target_count=0)
    print(f"Zero target: success={zero_result.success}")


def test_statistical_reporting():
    """Test the statistical reporting functionality."""
    validator = GPT2TokenValidator()
    
    print("\n=== Statistical Reporting Test ===")
    
    # Test batch validation
    word_categories = {
        "simple_nouns": ["cat", "dog", "house", "car"],
        "complex_nouns": ["smartphone", "neighborhood", "understanding", "intelligence"],
        "simple_verbs": ["run", "eat", "see", "go"],
        "complex_verbs": ["understand", "communicate", "demonstrate", "participate"]
    }
    
    batch_results = validator.batch_validate(word_categories, show_progress=False)
    
    print("Batch Validation Results:")
    print(f"Total categories: {batch_results['summary']['total_categories']}")
    print(f"Total words: {batch_results['summary']['total_words']}")
    print(f"Overall success rate: {batch_results['summary']['overall_success_rate']:.2f}")
    
    print("\nCategory Comparison:")
    for category, stats in batch_results['comparative_statistics']['category_comparison'].items():
        print(f"  {category}: {stats['valid_count']}/{stats['word_count']} ({stats['success_rate']:.2f})")
    
    print(f"Best category: {batch_results['comparative_statistics']['best_category']}")
    print(f"Worst category: {batch_results['comparative_statistics']['worst_category']}")
    
    # Test distribution analysis
    all_validation_results = []
    for results in batch_results['category_results'].values():
        all_validation_results.extend(results['validation_results'])
    
    distribution_report = validator.generate_distribution_report(all_validation_results)
    
    print(f"\nDistribution Analysis:")
    print(f"Token count distribution: {distribution_report['token_count_distribution']}")
    print(f"Most common token count: {distribution_report['most_common_token_count']}")
    print(f"Token count range: {distribution_report['token_count_range']}")
    print(f"Single-token words found: {len(distribution_report['complexity_analysis']['single_token_words'])}")
    print(f"Complex words found: {len(distribution_report['complexity_analysis']['complex_words'])}")


if __name__ == "__main__":
    test_validator()
    test_curation()
    test_statistical_reporting()