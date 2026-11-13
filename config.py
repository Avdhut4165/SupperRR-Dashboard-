"""Central configuration: constants, retention lists, exclusion rules."""
import re

# ---- modelling constants -------------------------------------------------
HP_QUANTILE = 0.75          # top 25% most consequential deliveries = high pressure
MIN_HP_BALLS = 100          # eligibility: high-pressure deliveries
MIN_HP_APPS = 15            # eligibility: high-pressure appearances
MIN_APP_BALLS = 3           # pressure balls needed for an "appearance"
CONS_PRIOR = 10             # beta prior strength for consistency
WEIGHTS = dict(L=0.60, C=0.25, D=0.15)   # level / consistency / resilience
BYE_RUNS = 1.0              # byes & leg-byes are not recorded -> assume 1 run
ASSOC_WEIGHT = 0.5          # weight for T20Is vs non-Full-Member opposition
N_FOLDS = 5
BOOT_SEED = 11
SPLIT_SEED = 7

FULL_MEMBERS = {'England', 'West Indies', 'India', 'Australia', 'Sri Lanka', 'South Africa',
                'Bangladesh', 'Zimbabwe', 'Afghanistan', 'Ireland', 'New Zealand', 'Pakistan'}

# ---- availability --------------------------------------------------------
# Approximate core-retention lists + confirmed trades supplied by the user (Nov 2026).
# Edit freely: any name here is removed from the ranking.
RETAINED = {
 'RCB': ['Virat Kohli', 'Rajat Patidar', 'Bhuvneshwar Kumar', 'Josh Hazlewood', 'Phil Salt', 'Jitesh Sharma', 'Krunal Pandya', 'Tim David'],
 'GT':  ['Shubman Gill', 'Sai Sudharsan', 'Kagiso Rabada', 'Rashid Khan', 'Jos Buttler', 'Mohammed Siraj'],
 'RR':  ['Vaibhav Sooryavanshi', 'Yashasvi Jaiswal', 'Dhruv Jurel', 'Riyan Parag', 'Ravindra Jadeja', 'Jofra Archer'],
 'SRH': ['Heinrich Klaasen', 'Ishan Kishan', 'Abhishek Sharma', 'Travis Head', 'Nitish Kumar Reddy', 'Natarajan',
         'Khaleel Ahmed', 'Nathan Ellis', 'Eshan Malinga'],
 'MI':  ['Jasprit Bumrah', 'Suryakumar Yadav', 'Tilak Varma', 'Rohit Sharma', 'Naman Dhir', 'Ryan Rickelton',
         'Trent Boult', 'Hardik Pandya'],
 'CSK': ['Ruturaj Gaikwad', 'Sanju Samson', 'Shivam Dube', 'Anshul Kamboj', 'Noor Ahmad', 'Dewald Brevis',
         'Liam Livingstone', 'Jack Edwards'],
 'DC':  ['KL Rahul', 'Axar Patel', 'Mitchell Starc', 'Rishabh Pant', 'Zeeshan Ansari'],
 'PBKS': ['Shreyas Iyer', 'Prabhsimran Singh', 'Priyansh Arya', 'Shashank Singh', 'Yuzvendra Chahal', 'Marco Jansen', 'Arshdeep Singh'],
 'KKR': ['Sunil Narine', 'Rinku Singh', 'Angkrish Raghuvanshi', 'Harshit Rana', 'Matheesha Pathirana', 'Varun Chakravarthy'],
 'LSG': ['Mitchell Marsh', 'Mohsin Khan', 'Prince Yadav', 'Aiden Markram', 'Kuldeep Yadav'],
}
CONDITIONAL = ['Cameron Green']        # KKR "keep or release": kept in the ranking, flagged

# Name used in the supplied lists -> name used in the ball-by-ball data
NAME_ALIASES = {'KL Rahul': 'Lokesh Rahul', 'Nitish Kumar Reddy': 'Nitish Reddy',
                'Natarajan': 'T Natarajan', 'Tilak Varma': 'Namboori Tilak Varma'}

# Retired (or retired from the IPL) -- matched by name
RETIRED = ['Dinesh Karthik', 'Andre Russell']

EXCLUDE_NATIONS = {
    'Pakistan': 'Pakistan national (not permitted in IPL)',
    'Bangladesh': 'Bangladesh national (BCCI forced KKR to release Mustafizur Jan 2026; none bought at IPL 2026 auction)',
}
INACTIVE_BEFORE = '2025-01-01'   # no IPL/T20I match in the data since this date => inactive

def norm(s: str) -> str:
    return re.sub(r'[^a-z ]', '', str(s).lower().replace('.', ' ')).strip()
