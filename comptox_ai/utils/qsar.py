import os
import shap
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score
from comptox_ai.db.graph_db import GraphDB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Data Retrieval and Formatting Functions

def grab_graph_data(listAcronym = "CPDAT"):
    db = GraphDB()
    print("Grabbed graph data...")
    return db.run_cypher('MATCH (cl:ChemicalList {listAcronym: "' + listAcronym + '"})-[r1]->(chem:Chemical)-[r2]->(a:Assay) RETURN chem.commonName AS chemName, chem.maccs AS maccs, r2 AS relation, a.commonName AS assayName')

def assign_assay_values(output):
    if pd.isna(output):
        return -1
    elif "CHEMICALHASACTIVEASSAY" in output:
        return 1
    else:
        return 0

def build_formatted_data(output):
    chemicals_df = pd.DataFrame(output)
    pivot_table = chemicals_df.pivot_table(index=['chemName', 'maccs'], columns='assayName', values='relation', aggfunc='first')
    for col in pivot_table.columns:
        pivot_table[col] = pivot_table[col].apply(lambda x: assign_assay_values(x))
    pivot_table.reset_index(inplace=True)
    return pivot_table

def maccs_key_conversion(maccs, human_readable = False):
    maccs_keys_descriptions = {
        0: 'Padding',
        1: 'ISOTOPE',
        2: 'limit the above def\'n since the RDKit only accepts up to #104',
        3: 'Group IVa,Va,VIa Rows 4-6',
        4: 'actinide',
        5: 'Group IIIB,IVB (Sc...)',
        6: 'Lanthanide',
        7: 'Group VB,VIB,VIIB',
        8: 'QAAA@1',
        9: 'Group VIII (Fe...)',
        10: 'Group IIa (Alkaline earth)',
        11: '4M Ring',
        12: 'Group IB,IIB (Cu..)',
        13: 'ON(C)C',
        14: 'S-S',
        15: 'OC(O)O',
        16: 'QAA@1',
        17: 'CTC',
        18: 'Group IIIA (B...)',
        19: '7M Ring',
        20: 'Si',
        21: 'C=C(Q)Q',
        22: '3M Ring',
        23: 'NC(O)O',
        24: 'N-O',
        25: 'NC(N)N',
        26: 'C$=C($A)$A',
        27: 'I',
        28: 'QCH2Q',
        29: 'P',
        30: 'CQ(C)(C)A',
        31: 'QX',
        32: 'CSN',
        33: 'NS',
        34: 'CH2=A',
        35: 'Group IA (Alkali Metal)',
        36: 'S Heterocycle',
        37: 'NC(O)N',
        38: 'NC(C)N',
        39: 'OS(O)O',
        40: 'S-O',
        41: 'CTN',
        42: 'F',
        43: 'QHAQH',
        44: 'OTHER',
        45: 'C=CN',
        46: 'BR',
        47: 'SAN',
        48: 'OQ(O)O',
        49: 'CHARGE',
        50: 'C=C(C)C',
        51: 'CSO',
        52: 'NN',
        53: 'QHAAAQH',
        54: 'QHAAQH',
        55: 'OSO',
        56: 'ON(O)C',
        57: 'O Heterocycle',
        58: 'QSQ',
        59: 'Snot%A%A',
        60: 'S=O',
        61: 'AS(A)A',
        62: 'A$!A$A',
        63: 'N=O',
        64: 'A$A!S',
        65: 'C%N',
        66: 'CC(C)(C)A',
        67: 'QS',
        68: 'QHQH (&...) SPEC Incomplete',
        69: 'QQH',
        70: 'QNQ',
        71: 'NO',
        72: 'OAAO',
        73: 'S=A',
        74: 'CH3ACH3',
        75: 'A!N$A',
        76: 'C=C(A)A',
        77: 'NAN',
        78: 'C=N',
        79: 'NAAN',
        80: 'NAAAN',
        81: 'SA(A)A',
        82: 'ACH2QH',
        83: 'QAAAA@1',
        84: 'NH2',
        85: 'CN(C)C',
        86: 'CH2QCH2',
        87: 'X!A$A',
        88: 'S',
        89: 'OAAAO',
        90: 'QHAACH2A',
        91: 'QHAAACH2A',
        92: 'OC(N)C',
        93: 'QCH3',
        94: 'QN',
        95: 'NAAO',
        96: '5 M ring',
        97: 'NAAAO',
        98: 'QAAAAA@1',
        99: 'C=C',
        100: 'ACH2N',
        101: '8M Ring or larger. This only handles up to ring sizes of 14',
        102: 'QO',
        103: 'CL',
        104: 'QHACH2A',
        105: 'A$A($A)$A',
        106: 'QA(Q)Q',
        107: 'XA(A)A',
        108: 'CH3AAACH2A',
        109: 'ACH2O',
        110: 'NCO',
        111: 'NACH2A',
        112: 'AA(A)(A)A',
        113: 'Onot%A%A',
        114: 'CH3CH2A',
        115: 'CH3ACH2A',
        116: 'CH3AACH2A',
        117: 'NAO',
        118: 'ACH2CH2A > 1',
        119: 'N=A',
        120: 'Heterocyclic atom > 1 (&...) Spec Incomplete',
        121: 'N Heterocycle',
        122: 'AN(A)A',
        123: 'OCO',
        124: 'QQ',
        125: 'Aromatic Ring > 1',
        126: 'A!O!A',
        127: 'A$A!O > 1 (&...) Spec Incomplete',
        128: 'ACH2AAACH2A',
        129: 'ACH2AACH2A',
        130: 'QQ > 1 (&...)  Spec Incomplete',
        131: 'QH > 1',
        132: 'OACH2A',
        133: 'A$A!N',
        134: 'X (HALOGEN)',
        135: 'Nnot%A%A',
        136: 'O=A>1',
        137: 'Heterocycle',
        138: 'QCH2A>1 (&...) Spec Incomplete',
        139: 'OH',
        140: 'O > 3 (&...) Spec Incomplete',
        141: 'CH3 > 2  (&...) Spec Incomplete',
        142: 'N > 1',
        143: 'A$A!O',
        144: 'Anot%A%Anot%A',
        145: '6M ring > 1',
        146: 'O > 2',
        147: 'ACH2CH2A',
        148: 'AQ(A)A',
        149: 'CH3 > 1',
        150: 'A!A$A!A',
        151: 'NH',
        152: 'OC(C)C',
        153: 'QCH2A',
        154: 'C=O',
        155: 'A!CH2!A',
        156: 'NA(A)A',
        157: 'C-O',
        158: 'C-N',
        159: 'O>1',
        160: 'CH3',
        161: 'N',
        162: 'Aromatic',
        163: '6M Ring',
        164: 'O',
        165: 'Ring',
        166: 'Fragments  FIX: this cant be done in SMARTS'
    }
    return [maccs_keys_descriptions[int(key)] for key in maccs] if human_readable else [maccs_keys_descriptions[int(key) - 1] for key in maccs]

def expand_maccs_column(datalist, human_readable = False):
    # maccs key is 167 rather than 166
    # Expand MACCS binary strings into separate columns
    maccs_expanded = datalist['maccs'].apply(lambda x: pd.Series(list(x)).astype(int))
    maccs_expanded.columns = maccs_key_conversion([i for i in range(maccs_expanded.shape[1])], human_readable=human_readable) if human_readable else [f'{i+1}' for i in range(maccs_expanded.shape[1])]
    datalist = datalist.drop(columns=['maccs'])
    final_df = pd.concat([datalist[['chemName']], maccs_expanded, datalist.drop(columns=['chemName'])], axis=1)
    return final_df

def makeQsarDataset(listAcronym="PFASMASTER", output_dir = "./output", human_readable=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    graphdata = grab_graph_data(listAcronym=listAcronym)
    cleaned_data = build_formatted_data(graphdata)
    formatted_data = expand_maccs_column(cleaned_data, human_readable = human_readable)
    formatted_data.to_csv(f"{output_dir.rstrip('/')}/comptox_{listAcronym}_all_assay.tsv", sep="\t")
    return formatted_data.set_index('chemName')

# Model Training Functions

def train_generic_model(clf, X, y, kwargs):
    # Scikit-like model interface
    clf = clf(**kwargs)
    return clf.fit(X, y)

def format_data_for_model(df, y_col_name, end_index = 167):
    X = df.iloc[:,:end_index]
    y = df[y_col_name]
    return X, y

def check_classes(y):
    check = False
    if (y.sum() == y.shape[0]) or (y.sum() == 0) or (min(y.value_counts()) < 4):
        check = True
    return check

def write_to_log(message, file_path, use_log=True):
    if use_log:
        with open(file_path, "a") as f:
            f.write(message + "\n")
    else:
        pass

def makeDiscoveryDataset(df, y_col_name, suffix = 1, output_dir = "./output"):
    if not os.path.exists(output_dir + "/discovery"):
        os.makedirs(output_dir + "/discovery")

    discovery_df = df[df[y_col_name] == -1] # Discovery set composed of chemicals without assay values
    discovery_df.to_csv(output_dir + "/discovery/discovery_model_" + str(suffix) + ".tsv", sep = "\t")
    return discovery_df

def makeDiscoveryDatasets(data, assays, output_dir = "./output"):
    if not os.path.exists(output_dir + "/discovery"):
        os.makedirs(output_dir + "/discovery")

    discovery_data = []
    for idx, assay in enumerate(assays):
        discovery_df = makeDiscoveryDataset(
            data, assay, suffix = idx, output_dir = output_dir)
        discovery_data.append(discovery_df)
        
    return discovery_data

def describe_dataset(data, y_col_name):
    X, y = data
    num_chemicals = X.shape[0]
    num_toxic = y[y == 1].shape[0]
    num_nontoxic = y[y == 0].shape[0]
    untested = y[y == -1].shape[0]
    return {'assay': y_col_name, 'num_chemicals': num_chemicals, 'num_toxic': num_toxic, 'num_nontoxic': num_nontoxic, 'untested': untested}

def describe_datasets(data, assays, output_dir = "./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_description = []
    for assay in assays:
        data_description.append(describe_dataset(format_data_for_model(data, assay), assay))
    dd_df = pd.DataFrame.from_dict(data_description)
    dd_df.to_csv(f"{output_dir}/data_description.tsv", sep = "\t")
    return dd_df

def trainQsarModel(
        df, y_col_name, clf, kwargs, suffix = 1, output_dir = "./output", seed = 42,
        save_model = True, save_discovery = True):
    # Expects that the first 167 columns are the macc columns
    focus_df = df[df[y_col_name] >= 0] # Only keep rows that have assay values
    discovery_df = df[df[y_col_name] == -1] # Discovery set composed of chemicals without assay values
    
    if not os.path.exists(output_dir + "/models"):
        os.makedirs(output_dir + "/models")

    # Format and Train the Model
    X, y = format_data_for_model(focus_df, y_col_name)

    if check_classes(y):
        return None, discovery_df, None
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=seed, stratify = y)

    # Train model and save
    model = train_generic_model(clf, train_X, train_y, kwargs)

    if save_model:
        pickle.dump(model, open(output_dir + "/models/model_" + str(suffix) + ".pkl", "wb"))

    # Test the model
    y_pred = model.predict(test_X)
    probs = model.predict_proba(test_X)[:, 1]

    rocauc = roc_auc_score(test_y, probs)
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    num_chemicals = test_X.shape[0]
    num_toxic = test_y.sum()
    num_nontoxic = num_chemicals - num_toxic
    row = {
        "assay": y_col_name,
        "rocauc": rocauc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_chemicals": num_chemicals,
        "num_toxic": num_toxic,
        "num_nontoxic": num_nontoxic
    }
    return model, test_X, row

def trainQsarModels(
        data, assays, clf, kwargs, output_dir = "./output", seed = None, remove_existing_output_folder = False,
        write_log = True, save_model = True, human_readable = False): 
    if os.path.exists(f"{output_dir}/log.txt") and remove_existing_output_folder:
        os.remove(f"{output_dir}/log.txt")
    
    if os.path.exists(f"{output_dir}/shap_plots") and remove_existing_output_folder:
        os.remove(f"{output_dir}/shap_plots")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(f"{output_dir}/shap_plots"):
        os.makedirs(f"{output_dir}/shap_plots")

    if not seed:
        seed = random.randint(0, 1000)

    models = []
    features = []
    evaluation = []
    skipped_models = 0
    data_description = []

    write_to_log(
        f"There are {len(assays)} assays and {data.shape[0]} chemicals that we are considering for this analysis:", f"{output_dir}/log.txt", write_log)
    for idx, assay in enumerate(assays):
        write_to_log(f"Training model for assay {assay} with seed {seed}.", f"{output_dir}/log.txt", write_log)
        model, X, row = trainQsarModel(
            data, assay, clf, kwargs, suffix=idx, seed = seed, output_dir = output_dir, save_model = save_model)
        if model is None or row is None:
            write_to_log(f"Warning: There are not enough of both classes (y) for assay {assay} to train a model.", f"{output_dir}/log.txt", write_log)
            skipped_models += 1
            continue
        models.append(model)
        features.append(X)
        evaluation.append(row)
        data_description.append(describe_dataset(format_data_for_model(data, assay), assay))

    for idx ,pair in enumerate(zip(models, features)):
        model, X = pair
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names = X.columns, show = False) if human_readable else shap.summary_plot(shap_values, X, feature_names = maccs_key_conversion(X.columns), show = False)
        plt.savefig(f"{output_dir}/shap_plots/shap_plot_{idx}.png")
        plt.clf()
    write_to_log(f"Saved {len(models)} models and skipped {skipped_models} assays due to insufficient data.", f"{output_dir}/log.txt", write_log)
    dd_df = pd.DataFrame.from_dict(data_description)
    dd_df.to_csv(f"{output_dir}/data_description.tsv", sep = "\t")
    return models, evaluation

# Model Evaluation Functions

def select_models(models, evaluation, by_rocauc = True, by_f1 = False, n = 10):
    target_col = 'rocauc' if by_rocauc else 'f1'
    best_models = evaluation.sort_values(by=[target_col], ascending=False)
    if n > 0:
        best_models = best_models.head(n=n)
    selected_models = []
    for idx, row in best_models.iterrows():
        selected_models.append(models[idx])
    return selected_models

def select_assays(evaluation, by_rocauc = True, by_f1 = False, n = 10):
    target_col = 'rocauc' if by_rocauc else 'f1'
    best_models = evaluation.sort_values(by=[target_col], ascending=False)
    if n > 0:
        best_models = best_models.head(n=n)
    return best_models['assay'].tolist()

def predictQsar(model, data, y_col_name, output_dir = "./output"):
    data = data[data[y_col_name] >= 0] # Only keep rows that have assay values
    X, y = format_data_for_model(data, y_col_name)
    y_pred = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    try:
        rocauc = roc_auc_score(y, probs)
    except:
        rocauc = 0
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    num_chemicals = X.shape[0]
    num_toxic = y.sum()
    num_nontoxic = num_chemicals - num_toxic
    return {
        "assay": y_col_name,
        "rocauc": rocauc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_chemicals": num_chemicals,
        "num_toxic": num_toxic,
        "num_nontoxic": num_nontoxic
    }

def validate_all_models(models, data, assays, output_dir = "./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each model, we check predictions on all assays
    validations = []
    for idx, model in enumerate(models):
        validations.append(predictQsar(model, data, assays[idx], output_dir = output_dir))
    return pd.DataFrame.from_dict(validations)

def display_results(results, toxic_cutoff = 1, sort_by = "f1"):
    return results[results['num_toxic'] > toxic_cutoff].sort_values(by=[sort_by], ascending=False)
