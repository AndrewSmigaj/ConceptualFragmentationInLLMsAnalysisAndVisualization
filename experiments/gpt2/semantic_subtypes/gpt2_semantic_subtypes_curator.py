"""
GPT-2 Semantic Subtypes Curator

Validates and curates single-token words for the semantic subtypes experiment.
Integrates the GPT2TokenValidator with the semantic word lists to produce
exactly 100 validated single-token words per semantic subtype.

Output: 800 validated words total (100 per subtype) ready for GPT-2 analysis.
"""

from gpt2_token_validator import GPT2TokenValidator
from gpt2_semantic_subtypes_wordlists import ALL_WORD_LISTS
import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SemanticSubtypesCurator:
    """Curates semantic subtypes word lists using GPT-2 token validation."""
    
    def __init__(self):
        """Initialize the curator with token validator."""
        self.validator = GPT2TokenValidator()
        self.target_words_per_subtype = 100
        
    def curate_all_subtypes(self, show_progress: bool = True) -> Dict:
        """
        Curate all 8 semantic subtypes to exactly 100 single-token words each.
        
        Args:
            show_progress: Whether to show progress information
            
        Returns:
            Dictionary containing curated words and statistics
        """
        if show_progress:
            print("=== GPT-2 Semantic Subtypes Curation ===")
            print(f"Target: {self.target_words_per_subtype} words per subtype")
            print(f"Total target: {len(ALL_WORD_LISTS) * self.target_words_per_subtype} words")
        
        curated_subtypes = {}
        all_statistics = {}
        
        for subtype_name, candidate_words in ALL_WORD_LISTS.items():
            if show_progress:
                print(f"\nCurating {subtype_name}: {len(candidate_words)} candidates")
            
            # Curate this subtype
            curation_result = self.validator.curate_wordlist(
                candidate_words, 
                target_count=self.target_words_per_subtype,
                show_progress=False
            )
            
            # Store results
            curated_subtypes[subtype_name] = curation_result.valid_words
            all_statistics[subtype_name] = {
                "candidates_available": len(candidate_words),
                "target_count": self.target_words_per_subtype,
                "achieved_count": curation_result.achieved_count,
                "success": curation_result.success,
                "efficiency": curation_result.statistics.get("efficiency", 0.0),
                "invalid_words_count": len(curation_result.invalid_words)
            }
            
            if show_progress:
                status = "SUCCESS" if curation_result.success else "FAILED"
                print(f"  {status}: {curation_result.achieved_count}/{self.target_words_per_subtype}")
                if not curation_result.success:
                    print(f"    Warning: Insufficient single-token words in {subtype_name}")
        
        # Calculate overall statistics
        total_words = sum(len(words) for words in curated_subtypes.values())
        successful_subtypes = sum(1 for stats in all_statistics.values() if stats["success"])
        
        overall_stats = {
            "total_subtypes": len(ALL_WORD_LISTS),
            "successful_subtypes": successful_subtypes,
            "total_words_curated": total_words,
            "target_total": len(ALL_WORD_LISTS) * self.target_words_per_subtype,
            "overall_success": successful_subtypes == len(ALL_WORD_LISTS),
            "average_efficiency": sum(stats["efficiency"] for stats in all_statistics.values()) / len(all_statistics)
        }
        
        if show_progress:
            print(f"\n=== Curation Summary ===")
            print(f"Successful subtypes: {successful_subtypes}/{len(ALL_WORD_LISTS)}")
            print(f"Total words curated: {total_words}/{len(ALL_WORD_LISTS) * self.target_words_per_subtype}")
            print(f"Average efficiency: {overall_stats['average_efficiency']:.2f}")
            print(f"Overall success: {overall_stats['overall_success']}")
        
        return {
            "curated_words": curated_subtypes,
            "subtype_statistics": all_statistics,
            "overall_statistics": overall_stats,
            "metadata": {
                "validator_used": "GPT2TokenValidator",
                "target_per_subtype": self.target_words_per_subtype,
                "source_lists": "gpt2_semantic_subtypes_wordlists.py"
            }
        }
    
    def save_curated_words(self, curation_results: Dict, output_path: str):
        """Save curated words to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(curation_results, f, indent=2, ensure_ascii=False)
        print(f"Curated words saved to: {output_path}")
    
    def get_validation_report(self, curation_results: Dict) -> str:
        """Generate a detailed validation report."""
        report_lines = [
            "=== GPT-2 Semantic Subtypes Validation Report ===",
            "",
            f"Target words per subtype: {self.target_words_per_subtype}",
            f"Total target words: {curation_results['overall_statistics']['target_total']}",
            f"Total achieved words: {curation_results['overall_statistics']['total_words_curated']}",
            f"Overall success: {curation_results['overall_statistics']['overall_success']}",
            "",
            "Per-subtype results:"
        ]
        
        for subtype, stats in curation_results['subtype_statistics'].items():
            status = "PASS" if stats["success"] else "FAIL"
            report_lines.append(
                f"  {status} {subtype:20} {stats['achieved_count']:3}/{stats['target_count']:3} "
                f"({stats['efficiency']:.2f} efficiency)"
            )
        
        report_lines.extend([
            "",
            f"Average efficiency: {curation_results['overall_statistics']['average_efficiency']:.3f}",
            "",
            "Sample words by subtype:"
        ])
        
        for subtype, words in curation_results['curated_words'].items():
            sample_words = words[:10]  # First 10 words as sample
            report_lines.append(f"  {subtype}: {', '.join(sample_words)}...")
        
        return "\n".join(report_lines)


def run_curation():
    """Run the complete curation process."""
    curator = SemanticSubtypesCurator()
    
    # Perform curation
    results = curator.curate_all_subtypes(show_progress=True)
    
    # Save results
    output_file = "gpt2_semantic_subtypes_curated.json"
    curator.save_curated_words(results, output_file)
    
    # Generate report
    report = curator.get_validation_report(results)
    print("\n" + report)
    
    # Save report
    report_file = "gpt2_semantic_subtypes_validation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nValidation report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    run_curation()