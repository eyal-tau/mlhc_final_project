import pandas as pd


# df = pd.read_csv("radiology_descriptions_classified.csv")
df = pd.read_csv("radiology_descriptions_classified_new.csv")

for i, r in df.iterrows():
    if r["is_invasive"] == "Yes" or r["is_invasive"] == "No":
        continue
    else:
        print("==================================================")
        print(f"{r['description']} - found bad classification - {r['is_invasive']}")
        if r["is_invasive_reason"] == "Yes" or r["tool_reason"] == "Yes":
            print(f"Found True - Yes verdict")
            df.at[i, "is_invasive"] = "Yes"
        elif r["is_invasive_reason"] == "No" or r["tool_reason"] == "No":
            print(f"Found True - No verdict")
            df.at[i, "is_invasive"] = "No"
        else:
            print("Missing verdict")
print("=========================================================")

df.to_csv("radiology_descriptions_classified_new_fixed.csv", index=False)


