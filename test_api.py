# test_api.py — финальная версия
import requests
import torch
import json
import sys
import os

# Добавляем путь к проекту
sys.path.insert(0, r"C:\anti\vscode\kiber_ml")

from dataset import get_or_generate_graph_data
from config import DATA_CFG

BASE_URL = "http://127.0.0.1:8000"

def print_separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def test_health():
    print_separator("HEALTH CHECK")
    r    = requests.get(f"{BASE_URL}/health")
    data = r.json()
    print(f"  Status:     {data['status'].upper()}")
    print(f"  Classes:    {data['n_classes']}")
    print(f"  Parameters: {data.get('parameters', 0):,}")
    print(f"  Device:     {data.get('device', 'N/A')}")
    print(f"  Threshold:  {data.get('threshold', 'N/A')}")

def test_real_incidents():
    """Тест на реальных примерах из датасета."""
    print_separator("REAL INCIDENT CLASSIFICATION")

    graph_data  = get_or_generate_graph_data()
    test_X      = graph_data["incident_X"]
    test_y      = graph_data["incident_y"]
    class_names = graph_data["class_names"]

    correct = 0
    total   = 0
    shown   = set()

    for i in range(len(test_X)):
        cls = test_y[i].item()
        if cls in shown:
            continue
        shown.add(cls)

        features = test_X[i].tolist()
        r = requests.post(
            f"{BASE_URL}/analyze",
            json={"features": features},
        )
        result = r.json()

        true_name = class_names[cls]
        pred_name = result["predicted_class"]
        conf      = result["confidence"]
        severity  = result["severity"].upper()
        alert     = result["alert"]
        ctx       = result["semantic_context"]
        match     = pred_name == true_name

        if match:
            correct += 1
        total += 1

        status = "OK  " if match else "MISS"
        alert_str = "ALERT" if alert else "     "

        print(
            f"  [{status}] [{alert_str}] "
            f"True: {true_name:<20s} "
            f"Pred: {pred_name:<20s} "
            f"conf={conf:.2f} "
            f"sev={severity:<8s}"
        )
        print(f"           KG context: {ctx}")

    print(f"\n  Result: {correct}/{total} correct "
          f"({correct/total*100:.0f}%)")

def test_explain_per_class():
    """XAI объяснение для трёх классов."""
    print_separator("XAI EXPLANATION (3 classes)")

    graph_data  = get_or_generate_graph_data()
    test_X      = graph_data["incident_X"]
    test_y      = graph_data["incident_y"]
    class_names = graph_data["class_names"]

    # Берём APT, Ransomware, DDoS
    target_classes = {"APT", "Ransomware", "DDoS"}
    shown = set()

    for i in range(len(test_X)):
        cls      = test_y[i].item()
        cls_name = class_names[cls]
        if cls_name not in target_classes or cls_name in shown:
            continue
        shown.add(cls_name)

        features = test_X[i].tolist()
        r = requests.post(
            f"{BASE_URL}/explain",
            json={"features": features},
        )
        result = r.json()

        print(f"\n  Incident type: {cls_name}")
        print(f"  Predicted:     {result['predicted_class']}")
        print(f"  Confidence:    {result['confidence']:.1%}")
        print(f"  Severity:      {result['severity'].upper()}")
        print(f"  KG Attention:")
        for item in result["explanation"]["top_attention"]:
            bar_len = int(item["weight"] * 40)
            bar     = "█" * bar_len + "░" * (40 - bar_len)
            print(f"    {item['class']:<20s} [{bar}] {item['weight']:.3f}")
        print(f"  → {result['explanation']['interpretation']}")

        if len(shown) == len(target_classes):
            break

def test_batch_stress():
    """Стресс-тест: 50 случайных запросов, считаем alert rate."""
    print_separator("STRESS TEST (50 random incidents)")

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    alert_count     = 0
    class_counts    = {}

    for i in range(50):
        features = torch.randn(128).tolist()
        r = requests.post(
            f"{BASE_URL}/analyze",
            json={"features": features},
        )
        result = r.json()

        sev  = result["severity"]
        pred = result["predicted_class"]
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        class_counts[pred]   = class_counts.get(pred, 0) + 1
        if result["alert"]:
            alert_count += 1

    print(f"\n  Severity distribution (random inputs):")
    for sev, cnt in severity_counts.items():
        bar = "█" * cnt
        print(f"    {sev:<10s} {bar:<50s} {cnt}")

    print(f"\n  Class distribution (random inputs):")
    for cls, cnt in sorted(class_counts.items(),
                           key=lambda x: x[1], reverse=True):
        bar = "█" * cnt
        print(f"    {cls:<20s} {bar} {cnt}")

    print(f"\n  Alert rate: {alert_count}/50 "
          f"({alert_count/50*100:.0f}%) — "
          f"ожидается низкий (случайные векторы)")

if __name__ == "__main__":
    print("\nCyberThreat Intelligence API — Test Suite")
    print("Ensure API is running: uvicorn api:app --port 8000\n")

    try:
        test_health()
        test_real_incidents()
        test_explain_per_class()
        test_batch_stress()
        print("\n\nAll tests complete.")

    except requests.exceptions.ConnectionError:
        print("ERROR: API not running.")
        print("Start it with: uvicorn api:app --reload --port 8000")