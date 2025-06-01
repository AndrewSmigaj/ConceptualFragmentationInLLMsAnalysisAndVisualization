"""
Test runner for Concept MRI.
Runs all test suites and generates a report.
"""
import unittest
import sys
import time
from pathlib import Path
from io import StringIO
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestReport:
    """Generate test report with detailed results."""
    
    def __init__(self):
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'suites': {},
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0
            }
        }
    
    def add_suite_results(self, suite_name: str, result: unittest.TestResult):
        """Add test suite results to report."""
        suite_data = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success': result.wasSuccessful(),
            'failed_tests': [],
            'error_tests': []
        }
        
        # Record failures
        for test, traceback in result.failures:
            suite_data['failed_tests'].append({
                'test': str(test),
                'traceback': traceback
            })
        
        # Record errors
        for test, traceback in result.errors:
            suite_data['error_tests'].append({
                'test': str(test),
                'traceback': traceback
            })
        
        self.results['suites'][suite_name] = suite_data
        
        # Update summary
        self.results['summary']['total_tests'] += result.testsRun
        self.results['summary']['passed'] += (
            result.testsRun - len(result.failures) - len(result.errors)
        )
        self.results['summary']['failed'] += len(result.failures)
        self.results['summary']['errors'] += len(result.errors)
    
    def generate_report(self, format='text'):
        """Generate test report in specified format."""
        if format == 'json':
            return json.dumps(self.results, indent=2)
        
        # Text format
        report = []
        report.append("=" * 70)
        report.append("CONCEPT MRI TEST REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append("")
        
        # Summary
        summary = self.results['summary']
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed']} ({summary['passed']/max(summary['total_tests'], 1)*100:.1f}%)")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Errors: {summary['errors']}")
        report.append("")
        
        # Suite details
        report.append("TEST SUITES")
        report.append("-" * 30)
        for suite_name, suite_data in self.results['suites'].items():
            status = "✓ PASSED" if suite_data['success'] else "✗ FAILED"
            report.append(f"\n{suite_name}: {status}")
            report.append(f"  Tests run: {suite_data['tests_run']}")
            
            if suite_data['failed_tests']:
                report.append(f"  Failures:")
                for failure in suite_data['failed_tests']:
                    report.append(f"    - {failure['test']}")
            
            if suite_data['error_tests']:
                report.append(f"  Errors:")
                for error in suite_data['error_tests']:
                    report.append(f"    - {error['test']}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def run_test_suite(suite_name: str, test_module):
    """Run a single test suite and return results."""
    print(f"\nRunning {suite_name}...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=1, stream=StringIO())
    result = runner.run(suite)
    
    return result


def main():
    """Run all test suites and generate report."""
    print("Starting Concept MRI Test Suite")
    print("=" * 50)
    
    report = TestReport()
    
    # Test suites to run
    test_suites = [
        ('Data Generators', 'concept_mri.tests.test_data_generators'),
        ('Component Tests', 'concept_mri.tests.test_components'),
        ('Integration Tests', 'concept_mri.tests.test_integration'),
    ]
    
    # Run each test suite
    for suite_name, module_name in test_suites:
        try:
            # Import test module
            test_module = __import__(module_name, fromlist=[''])
            
            # Run tests
            result = run_test_suite(suite_name, test_module)
            
            # Add to report
            report.add_suite_results(suite_name, result)
            
        except ImportError as e:
            print(f"  ERROR: Could not import {module_name}: {e}")
            # Add error to report
            error_result = unittest.TestResult()
            error_result.errors.append((module_name, str(e)))
            report.add_suite_results(suite_name, error_result)
        except Exception as e:
            print(f"  ERROR running {suite_name}: {e}")
            error_result = unittest.TestResult()
            error_result.errors.append((suite_name, str(e)))
            report.add_suite_results(suite_name, error_result)
    
    # Generate and display report
    print("\n" + "=" * 50)
    print(report.generate_report())
    
    # Save report to file
    report_file = Path(__file__).parent / 'test_report.txt'
    with open(report_file, 'w') as f:
        f.write(report.generate_report())
    
    # Save JSON report
    json_file = Path(__file__).parent / 'test_report.json'
    with open(json_file, 'w') as f:
        f.write(report.generate_report('json'))
    
    print(f"\nReports saved to:")
    print(f"  - {report_file}")
    print(f"  - {json_file}")
    
    # Return exit code based on results
    if report.results['summary']['failed'] > 0 or report.results['summary']['errors'] > 0:
        return 1
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)