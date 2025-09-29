#!/usr/bin/env python3
"""
Simple FAHN API connectivity test
"""

import requests
import socket

def test_dns_resolution():
    """Test if FAHN domain resolves"""
    try:
        ip = socket.gethostbyname("api.fahn.ai")
        print(f"DNS Resolution: api.fahn.ai -> {ip}")
        return True
    except socket.gaierror as e:
        print(f"DNS Resolution Failed: {e}")
        return False

def test_alternative_endpoints():
    """Test alternative FAHN endpoints"""
    endpoints = [
        "https://fahn.ai",
        "https://www.fahn.ai", 
        "https://api.fahn.com",
        "https://fahn.com",
        "https://fahn-api.com"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"{endpoint}: {response.status_code}")
        except Exception as e:
            print(f"{endpoint}: {str(e)}")

def main():
    print("FAHN API Connectivity Test")
    print("=" * 40)
    
    # Test DNS resolution
    dns_ok = test_dns_resolution()
    print()
    
    # Test alternative endpoints
    print("Testing alternative endpoints:")
    test_alternative_endpoints()
    print()
    
    if not dns_ok:
        print("CONCLUSION: FAHN API domain does not resolve")
        print("This suggests the API key may be for a different service")
        print("or the endpoint URL is incorrect")
    
    return dns_ok

if __name__ == "__main__":
    main()