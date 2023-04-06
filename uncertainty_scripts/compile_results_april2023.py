import pandas as pd
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

outputs_path = "/nasa/shared_homes/loic/kyodai-kmol/kmol_main2/kmol-internal/uncertainty_scripts/compiled_outputs"
logs_path = "/nasa/shared_homes/loic/kyodai-kmol/kmol_main2/kmol-internal/data/logs"

bc4_real_lrodd_preds = logs_path+"/lrodd_test_full/2023-04-04_14-48/predictions.csv"

chemblv2_path = "/nasa/datasets/kyodai_federated/proj_202111_202203/activity/prepared/chembl_v2_above_500.csv"
chemblv2_splits = "/nasa/datasets/kyodai_federated/proj_202111_202203/activity/prepared/splits.json"

exp_dict = {
    "bc4_real_lrodd": {"dataset_path": chemblv2_path, "splits_path": chemblv2_splits, "preds_path": bc4_real_lrodd_preds, "classification": True, "cols":["t_100n", "t_1u", "t_10u"]},
}


class PredictionProcessor:
    def __init__(self, all_experiments: dict):
        self.expriments = all_experiments

    def load_from_config(self, name: str, exp: dict): 
        self.exp_name = name
        self.dataset_df = pd.read_csv(exp["dataset_path"])
        self.predictions = pd.read_csv( exp["preds_path"])
        self.predictions.set_index('id', inplace=True)
        self.cols = exp["cols"]
        self.is_classification = exp["classification"]
        self.experiment_path = os.path.join(outputs_path,name) 
        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)

        if exp["splits_path"]:
            with open(exp["splits_path"]) as file:
                splits = json.load(file)
            self.test_df = self.dataset_df.iloc[splits['test']]
            self.train_df = self.dataset_df.iloc[splits['train']]
        elif 'target_sequence' in self.dataset_df.columns:
            self.test_df = self.dataset_df.iloc[self.predictions.index]
        
        self.predictions["protein_sequence"] = self.test_df["target_sequence"]
        self.predictions = self.process_predictions()
        self.per_prot_avg_df = self.predictions.groupby(["protein_sequence"]).mean()
        self.aggregation_mode = "sample"
        self.R_cols = {}

    def process_predictions(self,  threshold=0.5):
        self.predictions["cumulated_error"] = 0
        self.predictions["cumulated_logits_var"] = 0

        for col in self.cols:
            if self.is_classification:
                self.predictions[col+"_error"] = abs(self.predictions[col+"_ground_truth"] - (self.predictions[col] > threshold).astype(int))
            else:
                self.predictions[col+"_error"] = abs(self.predictions[col+"_ground_truth"] - self.predictions[col])
            
            self.predictions["cumulated_error"] += self.predictions[col+"_error"]

            if col+"_logits_var" in self.predictions.columns:
                self.predictions["cumulated_logits_var"] += self.predictions[col+"_logits_var"] #.abs()
            elif "likelihood_ratio" in self.predictions.columns:
                self.predictions[col+"_logits_var"] = self.predictions["likelihood_ratio"]
                self.predictions["cumulated_logits_var"] = self.predictions["likelihood_ratio"]
   
        return self.predictions
    
    def compute_metric(self, cols):
        gt_cols = [col+"_ground_truth" for col in cols]
        preds = self.predictions.loc[:, self.predictions.columns.isin(cols)].to_numpy().flatten()
        labels = self.predictions.loc[:, self.predictions.columns.isin(gt_cols)].to_numpy().flatten()

        if self.is_classification:
            accuracy = 1-(self.predictions["cumulated_error"].mean()/len(cols))

            overall_roc_auc = roc_auc_score(
                labels, preds
            )
            overall_avg_prec = average_precision_score(
                labels, preds
            )
            print(f"accuracy: {accuracy:.3f}")
            print(f"ROC-AUC: {overall_roc_auc:.3f}")
            print(f"Avg Precision: {overall_avg_prec:.3f}")

        else:
            l1 = self.predictions["cumulated_error"].mean()/len(self.cols)

            overall_rmse = np.sqrt(mean_squared_error(
                labels, preds
            ))

            overall_mae = mean_absolute_error(
                labels, preds
            )
            overall_r2 = r2_score(
                labels, preds
            )
            print(f"l1: {l1:2f}")
            print(f"RMSE: {overall_rmse:.3f}")
            print(f"MAE: {overall_mae:.3f}")
            print(f"R^2: {overall_r2:.3f}")

    def plot_unc_to_error(self, save_path="./mt_unc_to_error.png", col="t_100n", barplot=False):
        fig, ax = plt.subplots(figsize=(12,6))

        error = self.predictions[col+"_error"].values
        uncertainty = self.predictions[col+"_logits_var"].values  #.abs().values
        if barplot:
            
            bin_size = 0.1
            error_bins = {}
            for i, v in enumerate(uncertainty):
                bin_num = int(v/bin_size)
                if bin_num not in error_bins:
                    error_bins[bin_num] = []
                error_bins[bin_num].append(error[i])
            
            avg_bins = []
            count_in_bins = []

            sorted_keys  = sorted(error_bins.keys())
            
            for key in sorted_keys:
                avg_bins.append(np.mean(error_bins[key]))
                count_in_bins.append(len(error_bins[key]))

            X_plus = (np.asarray(list(sorted_keys))*bin_size)
            X_minus = (X_plus-bin_size).tolist()
            X_plus = X_plus.tolist()

            bin_names = []
            for i in range(len(X_plus)):
                bin_names.append(str(X_minus[i])[:3]+"-"+str(X_plus[i])[:3])
            
            cmap = mpl.cm.Blues(np.linspace(0,1,20))
            cmap = mpl.colors.ListedColormap(cmap[5:,:-1])

            sc = ax.bar(bin_names, avg_bins,  color=cmap(count_in_bins))

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(count_in_bins), vmax=max(count_in_bins)))
            sm.set_array(count_in_bins)
            cbar = fig.colorbar(sm)
            cbar.set_label("Count in bin")
                
        elif self.aggregation_mode != "sample":
            protein_count = self.predictions["protein_count"].values
            cmap = mpl.cm.Blues(np.linspace(0,1,20))
            cmap = mpl.colors.ListedColormap(cmap[5:,:-1])
            sc = ax.scatter(uncertainty, error, s=2, c=protein_count, cmap=cmap)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("protein count in trainset")
        else:
            ax.scatter(uncertainty, error, s=2)
        ax.set_xlabel("uncertainty")
        ax.set_ylabel("error")
        if col == 'cumulated':
            title = "epistemic uncertainty to error cumulated thresholds, R = "+str(self.R_cols[col])
        else:
            title = "epistemic uncertainty to error at threshold "+col+", R = "+str(self.R_cols[col])
        ax.set_title(title)
        fig.savefig(save_path)
        plt.close(fig)
    
    def plot_per_columns(self):
        for col in self.cols:
            self.plot_unc_to_error(save_path=os.path.join(self.experiment_path, self.aggregation_mode+"_"+col+".png"), col=col)
        self.plot_unc_to_error(save_path=os.path.join(self.experiment_path, self.aggregation_mode+"_cumulated.png"), col="cumulated")

    def compute_metrics_per_col(self):
        for col in self.cols:
            print(col)
            self.compute_metric([col])
        
    def compute_R_per_col(self, savepath=None):
        for col in self.cols:
            R_per_col = self.predictions[col+"_logits_var"].corr(self.predictions[col+"_error"]) #abs()
            print(col, "\nR_per_col = ", round(R_per_col, 3))
            self.R_cols[col] = round(R_per_col, 3)

        R_cum = self.predictions["cumulated_logits_var"].corr(self.predictions["cumulated_error"])            
        print("cumulated \nR_cumulated =", round(R_cum, 3))
        self.R_cols["cumulated"] = round(R_cum, 3)

        if savepath:
            with open(savepath, 'w') as outfile:
                for key, value in self.R_cols.items():
                    outfile.write(" & " + str(key))
                outfile.write("\n")

                for key, value in self.R_cols.items():
                    outfile.write( " & " + str(value))
                outfile.write("\n")

    def group_per_protein(self):
        return self.predictions.groupby(["protein_sequence"]).mean()

    def add_number_samples_in_trainset(self):
        protein_counts = self.train_df["target_sequence"].value_counts()
        self.predictions["protein_count"] = self.predictions.index.to_series().map(lambda prot: protein_counts.get(prot, 0))

    def get_protein_family(self):
        prot_df = pd.read_csv('/nasa/datasets/kyodai_federated/proj_202111_202203/activity/raw/protein-classification-all.csv')
        prot_df = prot_df.loc[:, ['pref_name', 'short_name', 'sequence']]

        self.predictions = self.predictions.reset_index()

        pref_names = []
        short_names = []

        for id in self.predictions.index:
            sequence = self.predictions.loc[id]["protein_sequence"]
            try:
                sequence_info = prot_df[prot_df["sequence"] == sequence].iloc[0]
            except IndexError:
                sequence_info = {"pref_name": "unknown", "short_name":"unknown"}
            pref_names.append(sequence_info["pref_name"])
            short_names.append(sequence_info["short_name"])

        self.predictions["pref_name"] = pref_names
        self.predictions["short_name"] = short_names

    def group_per_family(self):
        return self.predictions.groupby(["short_name"]).mean()

    def run(self):
        for name, exp in self.expriments.items():

            if name[:3] != "bc4":
                continue

            print("\nexp = ", name)
            print("\nper sample")
            self.load_from_config(name, exp)
            self.compute_metrics_per_col()
            self.compute_R_per_col(savepath=self.experiment_path+'/R_cols_sample.txt')
            self.plot_per_columns()

            if 'protein_sequence' in self.predictions.columns:
                self.aggregation_mode = "protein"
                print("\nper protein")
                self.predictions = self.group_per_protein()
                self.add_number_samples_in_trainset()
                self.compute_R_per_col(savepath=self.experiment_path+'/R_cols_protein.txt')
                self.plot_per_columns()
                if exp["classification"] == False:
                    self.plot_unc_to_error(save_path=os.path.join(self.experiment_path, self.aggregation_mode+"_barplot.png"), col=self.cols[0], barplot=True)

                print("\nper protein family")
                self.aggregation_mode = "protein_family"
                self.get_protein_family()
                self.predictions = self.group_per_family()
                self.compute_R_per_col(savepath=self.experiment_path+'/R_cols_protfamily.txt')
                self.plot_per_columns()
                if exp["classification"] == False:
                    self.plot_unc_to_error(save_path=os.path.join(self.experiment_path, self.aggregation_mode+"_barplot.png"), col=self.cols[0], barplot=True)

if __name__ == "__main__":
    predproc = PredictionProcessor(exp_dict)
    predproc.run()