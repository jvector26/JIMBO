"""Count Wayback captures (Sep 15 - opener) per season for candidate preseason minutes-projection pages."""
import csv, time, requests, os
C = ["hashtagbasketball.com/fantasy-basketball-projections", "www.fantasypros.com/nba/projections/", "basketballmonster.com/projections.aspx",
     "www.numberfire.com/nba/fantasy/remaining-projections", "www.rotowire.com/basketball/projections.php",
     "www.espn.com/fantasy/basketball/story/_/id", "www.cbssports.com/fantasy/basketball/stats/"]
rows = []
for pre in C:
    for S in range(2015, 2026):
        try:
            r = requests.get("http://web.archive.org/cdx/search/cdx", params={"url": pre, "matchType": "prefix", "output": "json",
                "from": f"{S}0915", "to": f"{S}1025", "filter": "statuscode:200", "fl": "timestamp,original,length"}, timeout=90)
            x = r.json()[1:] if r.status_code == 200 and r.text.strip() else []
        except Exception as e: x = []
        rows.append({"prefix": pre, "season": S, "n": len(x), "urls": " ".join(sorted({u[1] for u in x})[:5]), "last": x[-1][0] if x else ""})
        print(pre, S, len(x), flush=True); time.sleep(1.5)
os.makedirs("research/depth_wayback", exist_ok=True)
with open("research/depth_wayback/probe.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
