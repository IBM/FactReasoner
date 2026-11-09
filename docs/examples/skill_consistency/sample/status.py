"""Synthetic source for static analysis."""

import subprocess

import requests


def fetch_status():
    response = requests.get("https://example.com/status", timeout=10)
    subprocess.run(["git", "status"], check=True)
    return response.text
