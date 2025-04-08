import json
import random
from tqdm import tqdm  # optional, install with `pip install tqdm`

# === PROMPT GENERATORS ===

def generate_microeconomic_prompt():
    income = random.randint(30, 100)
    taxes = random.randint(5, 30)
    spend = income - taxes - random.randint(5, 20)
    return {
        "type": "micro",
        "style": "structured",
        "prompt": f"A household in the UK has income = {income} and expected taxes = {taxes}. What might they do?",
        "response": f"They might spend around {spend} and save the rest to smooth future consumption."
    }

def generate_macroeconomic_prompt():
    inflation = round(random.uniform(3.0, 9.0), 1)
    return {
        "type": "macro",
        "style": "structured",
        "prompt": f"The inflation rate in the US is {inflation}%. What action might the central bank take?",
        "response": "The Federal Reserve may raise interest rates to control inflation."
    }

def generate_policy_prompt():
    shock = random.choice(["Brexit", "COVID-19 lockdown", "global supply chain disruption"])
    return {
        "type": "policy",
        "style": "structured",
        "prompt": f"The economy experiences {shock}. What fiscal policy might be enacted?",
        "response": "The government may increase spending or provide stimulus payments to support households and businesses."
    }

def generate_casual_prompt():
    topic = random.choice([
        ("Why are food prices so high?",
         "It's likely inflation — general price increases across the economy."),
        ("I can't make ends meet this month. Any advice?",
         "Try cutting back on unnecessary spending like dining out or subscription services."),
        ("Why does the government keep raising taxes?",
         "It’s probably to pay for public services, but it could also be to balance the budget."),
        ("Why is petrol so expensive?",
         "It could be due to supply shortages or rising global oil prices.")
    ])
    return {
        "type": "casual",
        "style": "conversational",
        "prompt": topic[0],
        "response": topic[1]
    }

def generate_behavioral_prompt():
    fear = random.choice(["economic recession", "unemployment", "rising living costs"])
    response = random.choice(["save more", "avoid large purchases", "reduce spending on luxuries"])
    return {
        "type": "behavioral",
        "style": "structured",
        "prompt": f"If people fear {fear}, how might they behave?",
        "response": f"They may {response} to protect their finances."
    }

# === DATA GENERATOR ===

def generate_synthetic_data(n=50000, output_file="synthetic_econ_data.jsonl"):
    generators = [
        generate_microeconomic_prompt,
        generate_macroeconomic_prompt,
        generate_policy_prompt,
        generate_casual_prompt,
        generate_behavioral_prompt,
    ]

    with open(output_file, "w") as f:
        for _ in tqdm(range(n), desc="Generating synthetic examples"):
            sample = random.choice(generators)()
            f.write(json.dumps(sample) + "\n")

# === RUN ===

if __name__ == "__main__":
    generate_synthetic_data()
