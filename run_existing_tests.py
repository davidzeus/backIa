import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(filename='test_execution.log', encoding='utf-8', level=logging.DEBUG)

# Function to run a test file and capture the output

def run_test(test_file):
    try:
        result = subprocess.run(['python', test_file], check=True, capture_output=True, text=True)
        logging.info(f"{test_file} passed. Output:\n{result.stdout}")
        return (test_file, 'passed', result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"{test_file} failed. Error:\n{e.stderr}")
        return (test_file, 'failed', e.stderr)
    except Exception as e:
        logging.error(f"An exception occurred while running {test_file}: {str(e)}")
        return (test_file, 'error', str(e))

# List of test files to run

TestFiles = [
    'tests/test_vision_manual.py',
    'app/utils/test_temporal_parser.py'
]

# Run all tests and store results
results = []
missing_dependencies = []

for test_file in TestFiles:
    result = run_test(test_file)
    results.append(result)

# Report results

for test, status, output in results:
    print(f"Test: {test}, Status: {status}, Output: {output}")

# Check for missing dependencies (simplified check)
try:
    import pip
    installed_packages = {pkg.key for pkg in pip.get_installed_distributions()}
    required_packages = {'pytest', 'some-other-dependency'}  # Replace with actual dependencies

    missing_dependencies = required_packages - installed_packages
    if missing_dependencies:
        logging.warning(f"Missing dependencies: {', '.join(missing_dependencies)}")
except Exception as e:
    logging.error(f"Error while checking dependencies: {str(e)}")
