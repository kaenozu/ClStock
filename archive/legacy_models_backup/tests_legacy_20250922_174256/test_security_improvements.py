#!/usr/bin/env python3
"""
Test script for security and resource management improvements
"""

import requests
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_api_security():
    """Test API security features"""
    base_url = "http://localhost:8000/api/v1"

    print("Testing API security features...")

    # Test without authentication (should fail)
    try:
        response = requests.get(f"{base_url}/stocks")
        if response.status_code == 403:
            print("✓ Authentication required for /stocks endpoint")
        else:
            print(f"✗ Expected 403, got {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing unauthenticated access: {e}")

    # Test with valid API key
    headers = {"Authorization": "Bearer development-key"}
    try:
        response = requests.get(f"{base_url}/stocks", headers=headers)
        if response.status_code == 200:
            print("✓ Authenticated access successful")
        else:
            print(f"✗ Expected 200, got {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing authenticated access: {e}")

    # Test with invalid API key
    invalid_headers = {"Authorization": "Bearer invalid-key"}
    try:
        response = requests.get(f"{base_url}/stocks", headers=invalid_headers)
        if response.status_code == 401:
            print("✓ Invalid API key properly rejected")
        else:
            print(f"✗ Expected 401, got {response.status_code}")
    except Exception as e:
        print(f"✗ Error testing invalid API key: {e}")


def test_rate_limiting():
    """Test rate limiting functionality"""
    base_url = "http://localhost:8000/api/v1"
    headers = {"Authorization": "Bearer development-key"}

    print("\nTesting rate limiting...")

    # Make multiple requests to trigger rate limiting
    rate_limit_hit = False
    for i in range(10):
        try:
            response = requests.get(f"{base_url}/stocks", headers=headers)
            if response.status_code == 429:  # Too Many Requests
                print(f"✓ Rate limit triggered after {i+1} requests")
                rate_limit_hit = True
                break
            time.sleep(0.1)  # Small delay between requests
        except Exception as e:
            print(f"✗ Error during rate limit test: {e}")
            break

    if not rate_limit_hit:
        print(
            "ℹ Rate limit not triggered (may need more requests or different configuration)"
        )


def test_context_managers():
    """Test context manager functionality"""
    print("\nTesting context managers...")

    try:
        from utils.context_managers import managed_resource, file_lock
        from utils.temp_cleanup import create_temp_file

        # Test managed resource
        def create_dummy_resource():
            return "dummy_resource"

        def cleanup_dummy_resource(resource):
            pass  # Nothing to clean up for dummy resource

        with managed_resource(
            create_dummy_resource, cleanup_dummy_resource
        ) as resource:
            if resource == "dummy_resource":
                print("✓ Managed resource context manager works")
            else:
                print("✗ Managed resource context manager failed")

        # Test temporary file creation
        temp_file = create_temp_file(suffix=".txt", prefix="test_")
        if temp_file and "test_" in temp_file:
            print("✓ Temporary file creation works")
            # Cleanup will happen automatically
        else:
            print("✗ Temporary file creation failed")

        print("✓ Context manager tests completed")

    except Exception as e:
        print(f"✗ Error testing context managers: {e}")


def main():
    """Main test function"""
    print("ClStock Security and Resource Management Tests")
    print("=" * 50)

    test_context_managers()
    # Note: API tests require the server to be running
    # test_api_security()
    # test_rate_limiting()

    print(
        "\nTests completed. Note: API security tests require the server to be running."
    )


if __name__ == "__main__":
    main()
