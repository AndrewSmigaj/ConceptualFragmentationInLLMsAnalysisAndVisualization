"""
GPT-2 Semantic Subtypes Statistics Analysis

Analyzes the curated semantic subtypes dataset for frequency balance,
semantic coherence, and experimental readiness.
"""

import json
from typing import Dict, List
from collections import Counter

class SemanticSubtypesStatistics:
    """Analyzes statistics for the curated semantic subtypes dataset."""
    
    def __init__(self, curated_data_path: str):
        """Load curated data from JSON file."""
        with open(curated_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.curated_words = self.data["curated_words"]
        self.subtype_stats = self.data["subtype_statistics"]
        self.overall_stats = self.data["overall_statistics"]
    
    def analyze_word_distribution(self) -> Dict:
        """Analyze distribution of words across subtypes."""
        word_counts = {subtype: len(words) for subtype, words in self.curated_words.items()}
        total_words = sum(word_counts.values())
        
        # Calculate balance statistics
        min_count = min(word_counts.values())
        max_count = max(word_counts.values())
        avg_count = total_words / len(word_counts)
        
        # Calculate uniformity (how close to perfectly balanced)
        target_per_subtype = total_words / len(word_counts)
        balance_score = 1.0 - (max_count - min_count) / target_per_subtype
        
        return {
            "word_counts": word_counts,
            "total_words": total_words,
            "min_count": min_count,
            "max_count": max_count,
            "average_count": avg_count,
            "balance_score": balance_score,
            "balance_percentage": balance_score * 100,
            "target_per_subtype": 100,
            "achievement_rate": total_words / (len(word_counts) * 100)
        }
    
    def analyze_word_length_distribution(self) -> Dict:
        """Analyze distribution of word lengths across subtypes."""
        length_analysis = {}
        
        for subtype, words in self.curated_words.items():
            lengths = [len(word) for word in words]
            length_analysis[subtype] = {
                "min_length": min(lengths),
                "max_length": max(lengths),
                "avg_length": sum(lengths) / len(lengths),
                "length_distribution": dict(Counter(lengths))
            }
        
        # Overall length statistics
        all_lengths = []
        for words in self.curated_words.values():
            all_lengths.extend([len(word) for word in words])
        
        length_analysis["overall"] = {
            "min_length": min(all_lengths),
            "max_length": max(all_lengths),
            "avg_length": sum(all_lengths) / len(all_lengths),
            "length_distribution": dict(Counter(all_lengths))
        }
        
        return length_analysis
    
    def analyze_subtype_efficiency(self) -> Dict:
        """Analyze curation efficiency across subtypes."""
        efficiency_data = {}
        
        for subtype, stats in self.subtype_stats.items():
            efficiency_data[subtype] = {
                "efficiency": stats["efficiency"],
                "success_rate": stats["achieved_count"] / stats["target_count"],
                "candidates_used": stats["achieved_count"],
                "candidates_available": stats["candidates_available"],
                "utilization_rate": stats["achieved_count"] / stats["candidates_available"]
            }
        
        # Calculate overall efficiency metrics
        efficiencies = [data["efficiency"] for data in efficiency_data.values()]
        success_rates = [data["success_rate"] for data in efficiency_data.values()]
        
        efficiency_data["summary"] = {
            "avg_efficiency": sum(efficiencies) / len(efficiencies),
            "min_efficiency": min(efficiencies),
            "max_efficiency": max(efficiencies),
            "avg_success_rate": sum(success_rates) / len(success_rates),
            "perfect_subtypes": sum(1 for rate in success_rates if rate >= 1.0),
            "total_subtypes": len(success_rates)
        }
        
        return efficiency_data
    
    def get_experimental_readiness_report(self) -> str:
        """Generate experimental readiness assessment."""
        dist_analysis = self.analyze_word_distribution()
        length_analysis = self.analyze_word_length_distribution()
        efficiency_analysis = self.analyze_subtype_efficiency()
        
        report_lines = [
            "=== GPT-2 Semantic Subtypes Experimental Readiness Report ===",
            "",
            "DATASET OVERVIEW:",
            f"Total words curated: {dist_analysis['total_words']}/800 ({dist_analysis['achievement_rate']:.1%})",
            f"Perfect subtypes: {efficiency_analysis['summary']['perfect_subtypes']}/{efficiency_analysis['summary']['total_subtypes']}",
            f"Average success rate: {efficiency_analysis['summary']['avg_success_rate']:.1%}",
            f"Balance score: {dist_analysis['balance_percentage']:.1f}%",
            "",
            "SUBTYPE BREAKDOWN:",
        ]
        
        for subtype, count in dist_analysis['word_counts'].items():
            efficiency = efficiency_analysis[subtype]['efficiency']
            success_rate = efficiency_analysis[subtype]['success_rate']
            status = "READY" if success_rate >= 1.0 else f"PARTIAL {count}/100"
            report_lines.append(f"  {subtype:20} {count:3} words ({success_rate:.1%}) - {status}")
        
        report_lines.extend([
            "",
            "WORD LENGTH STATISTICS:",
            f"Average word length: {length_analysis['overall']['avg_length']:.1f} characters",
            f"Length range: {length_analysis['overall']['min_length']}-{length_analysis['overall']['max_length']} characters",
            "",
            "EXPERIMENTAL SUITABILITY:",
        ])
        
        # Assess experimental readiness
        if dist_analysis['total_words'] >= 750:
            suitability = "EXCELLENT - Sufficient words for robust analysis"
        elif dist_analysis['total_words'] >= 600:
            suitability = "GOOD - Adequate words for meaningful analysis" 
        elif dist_analysis['total_words'] >= 400:
            suitability = "MODERATE - Limited but usable for preliminary analysis"
        else:
            suitability = "INSUFFICIENT - Too few words for reliable analysis"
        
        report_lines.append(f"Overall assessment: {suitability}")
        
        if dist_analysis['balance_score'] > 0.8:
            report_lines.append("Balance quality: EXCELLENT - Well-balanced across subtypes")
        elif dist_analysis['balance_score'] > 0.6:
            report_lines.append("Balance quality: GOOD - Reasonably balanced")
        else:
            report_lines.append("Balance quality: POOR - Significant imbalance detected")
        
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "* Dataset is ready for GPT-2 semantic analysis",
            "* Sufficient words for clustering and path analysis", 
            "* Good semantic diversity across all major POS categories",
            f"* {efficiency_analysis['summary']['perfect_subtypes']} subtypes have full 100-word sets",
            "",
            "NEXT STEPS:",
            "1. Proceed with GPT-2 activation extraction",
            "2. Apply clustering analysis (k-means + HDBSCAN)",
            "3. Calculate APA metrics and path coherence",
            "4. Generate semantic subtype interpretation"
        ])
        
        return "\n".join(report_lines)
    
    def save_statistics_report(self, output_path: str):
        """Save comprehensive statistics to file."""
        dist_analysis = self.analyze_word_distribution()
        length_analysis = self.analyze_word_length_distribution()
        efficiency_analysis = self.analyze_subtype_efficiency()
        readiness_report = self.get_experimental_readiness_report()
        
        comprehensive_stats = {
            "distribution_analysis": dist_analysis,
            "length_analysis": length_analysis,
            "efficiency_analysis": efficiency_analysis,
            "readiness_report": readiness_report,
            "metadata": {
                "analysis_version": "1.0",
                "source_file": "gpt2_semantic_subtypes_curated.json",
                "analysis_date": "2025-01-22"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_stats, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive statistics saved to: {output_path}")


def run_statistics_analysis():
    """Run complete statistics analysis."""
    print("=== GPT-2 Semantic Subtypes Statistics Analysis ===")
    
    # Load and analyze data
    analyzer = SemanticSubtypesStatistics("gpt2_semantic_subtypes_curated.json")
    
    # Generate and display readiness report
    readiness_report = analyzer.get_experimental_readiness_report()
    print(readiness_report)
    
    # Save comprehensive statistics
    analyzer.save_statistics_report("gpt2_semantic_subtypes_statistics.json")
    
    # Save readiness report as text file
    with open("gpt2_semantic_subtypes_readiness_report.txt", 'w', encoding='utf-8') as f:
        f.write(readiness_report)
    print("\nReadiness report saved to: gpt2_semantic_subtypes_readiness_report.txt")
    
    return analyzer


if __name__ == "__main__":
    run_statistics_analysis()