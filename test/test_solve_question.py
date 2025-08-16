import sys
import os
import asyncio
from fastapi.testclient import TestClient
from main import app, SolveQuestionRequest, MathChains
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize test client
client = TestClient(app)

# Test question
TEST_QUESTION = """
设函数 f:(a,b)→R 在开区间 (a,b) 上一致连续。证明 f 在 (a,b) 上有界。
"""

async def test_solve_question():
    """Test the solve_question function with a mathematical problem."""
    try:
        # Create a request
        request = SolveQuestionRequest(question=TEST_QUESTION)
        
        # Call the endpoint
        response = client.post("/api/solve_question", json=request.dict())
        
        # Check response status code
        assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
        
        # Parse response
        result = response.json()
        
        # Print the results
        print("\n=== Test Results ===")
        print(f"Status: Success")
        print(f"Request ID: {result.get('request_id')}")
        print(f"Processing Time: {result.get('processing_time')}s")
        print("\n=== Detailed Solution ===")
        print(result.get('detailed_solution', 'No solution provided'))
        
        return True
        
    except Exception as e:
        print(f"\n=== Test Failed ===\nError: {str(e)}")
        return False

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set")
        sys.exit(1)
    
    # Run the test
    success = asyncio.run(test_solve_question())
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
