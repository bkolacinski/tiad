import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Zadanie 3 — Klasyfikacja obrazów (Cards Image Dataset)

        Porównanie 5 modeli CNN (transfer learning z ImageNet) na 53 klasach kart,
        dla 4 podziałów train/test. Metryki: accuracy, precision, recall, F1, ROC/AUC.
        """
    )
    return


@app.cell
def _():
    import os
    import site
    import glob

    nvidia_paths = glob.glob(os.path.join(site.getsitepackages()[0], "nvidia", "*", "lib"))
    if nvidia_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import tensorflow as tf
    import numpy as np
    import pandas as pd

    gpus = tf.config.list_physical_devices("GPU")
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g, True)

    print(f"TF {tf.__version__}  |  GPU: {gpus}")
    return gpus, np, pd, tf


@app.cell
def _():
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    MODELS_DIR = "models"

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 8
    EPOCHS_PHASE2 = 12
    LR_PHASE1 = 1e-3
    LR_PHASE2 = 1e-5
    UNFREEZE_FRACTION = 1.0 / 3.0
    EARLY_STOP_PATIENCE = 5
    SEED = 42

    SPLITS = [0.50, 0.60, 0.70, 0.80, 0.90]

    MODEL_NAMES = ["MobileNetV2", "ResNet50", "EfficientNetB0", "InceptionV3", "VGG16"]

    SMOKE_TEST = False
    return (
        BATCH_SIZE,
        DATA_DIR,
        EARLY_STOP_PATIENCE,
        EPOCHS_PHASE1,
        EPOCHS_PHASE2,
        IMAGE_SIZE,
        LR_PHASE1,
        LR_PHASE2,
        MODELS_DIR,
        MODEL_NAMES,
        RESULTS_DIR,
        SEED,
        SMOKE_TEST,
        SPLITS,
        UNFREEZE_FRACTION,
    )


@app.cell
def _(DATA_DIR, MODELS_DIR, RESULTS_DIR):
    import os as _os
    for d in (RESULTS_DIR, MODELS_DIR):
        _os.makedirs(d, exist_ok=True)
    return


@app.cell
def _(DATA_DIR, pd):
    import os as _os
    df = pd.read_csv(f"{DATA_DIR}/cards.csv")
    df["full_path"] = DATA_DIR + "/" + df["filepaths"]
    df = df[df["full_path"].map(_os.path.isfile)].reset_index(drop=True)
    class_names = sorted(df["card type"].unique())
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    df["label_idx"] = df["card type"].map(label_to_idx)
    NUM_CLASSES = len(class_names)
    print(f"{NUM_CLASSES} classes, {len(df)} images")
    print(df["data set"].value_counts())
    return NUM_CLASSES, class_names, df, label_to_idx


@app.cell
def _(df, mo):
    mo.ui.table(df.head(8))
    return


@app.cell
def _(SEED, df, np):
    from sklearn.model_selection import train_test_split

    def make_split(train_frac):
        all_paths = df["full_path"].to_numpy()
        all_labels = df["label_idx"].to_numpy()
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_paths,
            all_labels,
            train_size=train_frac,
            stratify=all_labels,
            random_state=SEED,
        )
        return train_paths, train_labels, test_paths, test_labels

    return (make_split,)


@app.cell
def _(BATCH_SIZE, IMAGE_SIZE, NUM_CLASSES, tf):
    AUTOTUNE = tf.data.AUTOTUNE

    def _decode(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        return img, label

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augment",
    )

    def build_dataset(paths, labels, preprocess_fn, training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        if training:
            ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
        return ds.prefetch(AUTOTUNE)

    return AUTOTUNE, build_dataset


@app.cell
def _(IMAGE_SIZE, NUM_CLASSES, tf):
    from tensorflow.keras import applications as apps

    MODEL_REGISTRY = {
        "MobileNetV2": (apps.MobileNetV2, apps.mobilenet_v2.preprocess_input),
        "ResNet50": (apps.ResNet50, apps.resnet50.preprocess_input),
        "EfficientNetB0": (apps.EfficientNetB0, apps.efficientnet.preprocess_input),
        "InceptionV3": (apps.InceptionV3, apps.inception_v3.preprocess_input),
        "VGG16": (apps.VGG16, apps.vgg16.preprocess_input),
    }

    def build_model(name):
        ctor, preprocess = MODEL_REGISTRY[name]
        backbone = ctor(
            include_top=False,
            weights="imagenet",
            input_shape=IMAGE_SIZE + (3,),
            pooling="avg",
        )
        backbone.trainable = False

        inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
        x = backbone(inputs, training=False)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name=name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model, preprocess

    return MODEL_REGISTRY, build_model


@app.cell
def _(
    EARLY_STOP_PATIENCE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LR_PHASE1,
    LR_PHASE2,
    UNFREEZE_FRACTION,
    build_dataset,
    build_model,
    tf,
):
    import time

    class _Hist:
        def __init__(self, h):
            self.history = h

    def _merge_history(h1, h2):
        keys = set(h1.history) | set(h2.history)
        return _Hist({k: list(h1.history.get(k, [])) + list(h2.history.get(k, [])) for k in keys})

    def train_one(model_name, train_paths, train_labels, test_paths, test_labels):
        model, preprocess = build_model(model_name)
        train_ds = build_dataset(train_paths, train_labels, preprocess, training=True)
        test_ds = build_dataset(test_paths, test_labels, preprocess, training=False)

        cb1 = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
        t0 = time.time()
        h1 = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_PHASE1, callbacks=cb1, verbose=2)

        backbone = model.layers[1]
        backbone.trainable = True
        n_layers = len(backbone.layers)
        n_unfreeze = max(1, int(n_layers * UNFREEZE_FRACTION))
        for layer in backbone.layers[:-n_unfreeze]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        cb2 = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
        h2 = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_PHASE2, callbacks=cb2, verbose=2)

        dt = time.time() - t0
        return model, _merge_history(h1, h2), test_ds, dt

    return (train_one,)


@app.cell
def _(NUM_CLASSES, np, tf):
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_auc_score,
        roc_curve,
    )

    def evaluate(model, test_ds):
        y_true = np.concatenate([y.numpy() for _, y in test_ds])
        y_prob = model.predict(test_ds, verbose=0)
        y_pred = y_prob.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        y_true_oh = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)
        try:
            auc_macro = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
            auc_micro = roc_auc_score(y_true_oh, y_prob, average="micro", multi_class="ovr")
        except ValueError:
            auc_macro = float("nan")
            auc_micro = float("nan")

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": cm,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "auc_macro": auc_macro,
            "auc_micro": auc_micro,
            "report": report,
        }

    return (evaluate,)


@app.cell
def _(MODEL_NAMES, RESULTS_DIR, SMOKE_TEST, SPLITS, evaluate, make_split, np, pd, tf, train_one):
    import os as _os
    import json as _json
    import gc as _gc

    _runs = [("ResNet50", 0.80)] if SMOKE_TEST else [(_m, _s) for _m in MODEL_NAMES for _s in SPLITS]

    csv_path = _os.path.join(RESULTS_DIR, "metrics.csv")
    history_dir = _os.path.join(RESULTS_DIR, "history")
    cm_dir = _os.path.join(RESULTS_DIR, "cm")
    prob_dir = _os.path.join(RESULTS_DIR, "probs")
    for _d in (history_dir, cm_dir, prob_dir):
        _os.makedirs(_d, exist_ok=True)

    if _os.path.exists(csv_path):
        _rows = pd.read_csv(csv_path).to_dict("records")
        _done = {(_r["model"], float(_r["split"])) for _r in _rows}
    else:
        _rows = []
        _done = set()

    for _model_name, _split in _runs:
        if (_model_name, _split) in _done:
            print(f"skip {_model_name} {_split} (cached)")
            continue
        _tp, _tl, _vp, _vl = make_split(_split)
        print(f"\n=== {_model_name}  split={_split:.2f}  train={len(_tp)}  test={len(_vp)} ===")
        _model, _history, _test_ds, _dt = train_one(_model_name, _tp, _tl, _vp, _vl)
        _res = evaluate(_model, _test_ds)

        _tag = f"{_model_name}_{int(_split*100):02d}"
        np.save(_os.path.join(cm_dir, f"{_tag}.npy"), _res["cm"])
        np.save(_os.path.join(prob_dir, f"{_tag}_y_true.npy"), _res["y_true"])
        np.save(_os.path.join(prob_dir, f"{_tag}_y_prob.npy"), _res["y_prob"])
        with open(_os.path.join(history_dir, f"{_tag}.json"), "w") as _f:
            _json.dump({_k: [float(_v) for _v in _vs] for _k, _vs in _history.history.items()}, _f)

        _rows.append({
            "model": _model_name,
            "split": _split,
            "accuracy": _res["accuracy"],
            "macro_f1": _res["macro_f1"],
            "macro_precision": _res["macro_precision"],
            "macro_recall": _res["macro_recall"],
            "auc_macro": _res["auc_macro"],
            "auc_micro": _res["auc_micro"],
            "epochs_run": len(_history.history["loss"]),
            "train_time_s": _dt,
        })
        pd.DataFrame(_rows).to_csv(csv_path, index=False)
        print(f"  acc={_res['accuracy']:.4f}  f1={_res['macro_f1']:.4f}  auc={_res['auc_macro']:.4f}  time={_dt:.1f}s")

        del _model, _history, _test_ds, _res
        tf.keras.backend.clear_session()
        _gc.collect()

    metrics_df = pd.DataFrame(_rows)
    return cm_dir, csv_path, history_dir, metrics_df, prob_dir


@app.cell
def _(metrics_df, mo):
    mo.md("## Tabela zbiorcza")
    return


@app.cell
def _(metrics_df):
    summary = metrics_df.copy()
    summary = summary[["model", "split", "accuracy", "macro_f1", "macro_precision",
                       "macro_recall", "auc_macro", "epochs_run", "train_time_s"]]
    summary = summary.sort_values(["model", "split"]).reset_index(drop=True)
    summary
    return (summary,)


@app.cell
def _(metrics_df, mo):
    pivot_acc = metrics_df.pivot(index="model", columns="split", values="accuracy")
    pivot_f1 = metrics_df.pivot(index="model", columns="split", values="macro_f1")
    pivot_auc = metrics_df.pivot(index="model", columns="split", values="auc_macro")
    mo.md("### Accuracy / F1 / AUC × model × split")
    return pivot_acc, pivot_auc, pivot_f1


@app.cell
def _(pivot_acc, pivot_auc, pivot_f1):
    print("ACCURACY")
    print(pivot_acc.round(4))
    print("\nMACRO F1")
    print(pivot_f1.round(4))
    print("\nAUC MACRO")
    print(pivot_auc.round(4))
    return


@app.cell
def _(RESULTS_DIR, metrics_df):
    import matplotlib.pyplot as plt
    import os as _os

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, _metric, _title in zip(
        axes,
        ["accuracy", "macro_f1", "auc_macro"],
        ["Accuracy", "Macro F1", "Macro AUC"],
    ):
        for _name, _group in metrics_df.groupby("model"):
            _gg = _group.sort_values("split")
            _ax.plot(_gg["split"], _gg[_metric], marker="o", label=_name)
        _ax.set_xlabel("Train fraction")
        _ax.set_ylabel(_metric)
        _ax.set_title(_title)
        _ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    _out = _os.path.join(RESULTS_DIR, "metrics_vs_split.png")
    plt.savefig(_out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"saved {_out}")
    return axes, fig, plt


@app.cell
def _(RESULTS_DIR, class_names, cm_dir, metrics_df, np, plt):
    import os as _os
    import seaborn as sns

    best = metrics_df.loc[metrics_df["accuracy"].idxmax()]
    best_tag = f"{best['model']}_{int(best['split']*100):02d}"
    cm_best = np.load(_os.path.join(cm_dir, f"{best_tag}.npy"))

    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_best, cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True, ax=ax_cm)
    ax_cm.set_title(f"Confusion matrix — {best['model']} (split {best['split']:.2f})  acc={best['accuracy']:.3f}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    out_cm = _os.path.join(RESULTS_DIR, f"cm_{best_tag}.png")
    plt.savefig(out_cm, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"best: {best['model']} @ split {best['split']:.2f}  acc={best['accuracy']:.4f}")
    return ax_cm, best, best_tag, cm_best, fig_cm, out_cm, sns


@app.cell
def _(NUM_CLASSES, RESULTS_DIR, best, best_tag, class_names, np, plt, prob_dir):
    import os as _os
    from sklearn.metrics import roc_curve as _roc_curve, auc as _auc

    y_true = np.load(_os.path.join(prob_dir, f"{best_tag}_y_true.npy"))
    y_prob = np.load(_os.path.join(prob_dir, f"{best_tag}_y_prob.npy"))

    per_class_auc = []
    for _c in range(NUM_CLASSES):
        _y_bin = (y_true == _c).astype(int)
        if _y_bin.sum() == 0:
            per_class_auc.append((_c, np.nan))
            continue
        _fpr, _tpr, _ = _roc_curve(_y_bin, y_prob[:, _c])
        per_class_auc.append((_c, _auc(_fpr, _tpr)))

    _aucs_sorted = sorted(per_class_auc, key=lambda x: (np.nan_to_num(x[1], nan=0)))
    _picked = _aucs_sorted[:5] + _aucs_sorted[-5:]

    fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
    for _c, _a in _picked:
        _y_bin = (y_true == _c).astype(int)
        _fpr, _tpr, _ = _roc_curve(_y_bin, y_prob[:, _c])
        ax_roc.plot(_fpr, _tpr, label=f"{class_names[_c]} (AUC={_a:.3f})", lw=1.2)
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title(f"ROC OvR — {best['model']} (split {best['split']:.2f})\n5 najgorszych + 5 najlepszych klas")
    ax_roc.legend(loc="lower right", fontsize=7)
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    out_roc = _os.path.join(RESULTS_DIR, f"roc_{best_tag}.png")
    plt.savefig(out_roc, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"saved {out_roc}")
    return ax_roc, fig_roc, out_roc, per_class_auc, y_prob, y_true


@app.cell
def _(MODEL_NAMES, RESULTS_DIR, SPLITS, history_dir, plt):
    import os as _os
    import json as _json

    fig_h, axes_h = plt.subplots(len(MODEL_NAMES), 2, figsize=(11, 2.6 * len(MODEL_NAMES)))
    if len(MODEL_NAMES) == 1:
        axes_h = axes_h.reshape(1, 2)
    for _i, _name in enumerate(MODEL_NAMES):
        for _split in SPLITS:
            _tag = f"{_name}_{int(_split*100):02d}"
            _p = _os.path.join(history_dir, f"{_tag}.json")
            if not _os.path.exists(_p):
                continue
            with open(_p) as _f:
                _h = _json.load(_f)
            _label = f"split {_split:.2f}"
            axes_h[_i, 0].plot(_h["loss"], label=f"train {_label}", alpha=0.7)
            axes_h[_i, 0].plot(_h["val_loss"], label=f"val {_label}", linestyle="--", alpha=0.7)
            axes_h[_i, 1].plot(_h["accuracy"], label=f"train {_label}", alpha=0.7)
            axes_h[_i, 1].plot(_h["val_accuracy"], label=f"val {_label}", linestyle="--", alpha=0.7)
        axes_h[_i, 0].set_title(f"{_name} — loss")
        axes_h[_i, 1].set_title(f"{_name} — accuracy")
        axes_h[_i, 0].grid(True, alpha=0.3)
        axes_h[_i, 1].grid(True, alpha=0.3)
        axes_h[_i, 0].legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    out_h = _os.path.join(RESULTS_DIR, "learning_curves.png")
    plt.savefig(out_h, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"saved {out_h}")
    return axes_h, fig_h, out_h


if __name__ == "__main__":
    app.run()
