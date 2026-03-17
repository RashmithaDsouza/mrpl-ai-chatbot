#!/bin/bash

ollama serve &

sleep 5

ollama pull phi3

streamlit run App.py --server.port=10000 --server.address=0.0.0.0
