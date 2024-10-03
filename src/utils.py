import json

def make_report(metrics:dict) -> None:
    with open('report_metrics.json', 'w') as report:
        json.dump(metrics, report)
