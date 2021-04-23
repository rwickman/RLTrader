import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
import numpy as np
sns.set(style="darkgrid", font_scale=1.5)

with open("../models/train_dict.json") as f:
    train_dict = json.load(f)


fig, axs = plt.subplots(2)

val_ep_str = [str(x+1) for x in list(range(len(train_dict["val_balance"])))]
sns.lineplot(x=val_ep_str, y=train_dict["val_balance"], ax=axs[0])
axs[0].set(xlabel="Validation Episode", ylabel="Final Balance")

loss_df = pd.DataFrame(np.array(train_dict["train_losses"]).T).melt()
sns.lineplot(data=loss_df, x="variable", y="value", ax=axs[1])
axs[1].set(xlabel="Episode", ylabel="Train Loss")

#
# plt.plot(train_dict["val_balance"])
plt.show()