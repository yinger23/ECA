from preprocess import *
import pandas as pd
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 计算F-β分数
def f_beta_score(precision, recall, beta):
    # 检查分母是否为0
    denominator = (beta ** 2 * precision + recall)
    if denominator == 0:
        return 0  # 如果分母为0，返回0（或其他默认值）
    return (1 + beta ** 2) * (precision * recall) / denominator

def plot_results(history, y_test, y_pred_prob):
    """绘制训练曲线和ROC曲线"""
    plt.figure(figsize=(15, 5))

    # 训练和验证损失
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 训练和验证准确率
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # ROC曲线
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.tight_layout()
    plt.show()

def MLP_CAT(X_train, X_test, y_train, y_test, beta=1):
    # 获取输入特征的维度
    input_dim = X_train.shape[1]

    # 创建模型
    model = Sequential()
    # 输入层和第一个隐藏层
    model.add(Dense(64, input_dim=input_dim, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.3))  # Dropout 层，防止过拟合
    # 第二个隐藏层
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    # 第三个隐藏层
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.2))
    # 输出层（二分类问题，使用 sigmoid 激活函数）
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # 打印模型摘要
    model.summary()

    # 训练模型
    # 早停（EarlyStopping）技术
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=300,  # 训练轮数
        batch_size=32,  # 批量大小
        validation_split=0.2,  # 验证集比例
        verbose=1,  # 显示训练过程
        callbacks=[early_stopping]
    )

    # 预测概率
    y_pred_prob = model.predict(X_test)

    # 计算ROC曲线和AUC值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # 初始化变量
    best_threshold = 0
    best_f_beta = 0

    # 遍历所有阈值，计算F-β分数
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f_beta = f_beta_score(precision, recall, beta)

        # 更新最佳阈值
        if f_beta > best_f_beta:
            best_f_beta = f_beta
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F-{beta} Score: {best_f_beta:.4f}")

    # 使用最佳阈值进行最终预测
    y_pred_best = (y_pred_prob >= best_threshold).astype(int)

    # 计算最终评估指标
    accuracy = accuracy_score(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best)
    recall = recall_score(y_test, y_pred_best)
    f1 = f1_score(y_test, y_pred_best)

    print(f"Accuracy (Best Threshold): {accuracy:.4f}")
    print(f"Precision (Best Threshold): {precision:.4f}")
    print(f"Recall (Best Threshold): {recall:.4f}")
    print(f"F1 Score (Best Threshold): {f1:.4f}")

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 绘制训练和验证的损失曲线
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制训练和验证的准确率曲线
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def MLP_CAT_fold(folds, beta=1, epochs=500, batch_size=32):
    # 初始化存储每折指标的字典
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'optimal_threshold': [],
        'best_f_beta': []
    }

    best_auc = -1
    best_fold_data = None  # 保存最佳折的数据

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        # 获取输入特征的维度
        input_dim = X_train.shape[1]

        # 进一步划分验证集（从训练集划分）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # 创建模型
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='tanh', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='tanh'),
            Dropout(0.3),
            Dense(16, activation='tanh'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        # 编译模型
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 训练模型（使用验证集早停）
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[early_stopping]
        )
        # 预测概率
        y_pred_prob = model.predict(X_test)
        # 计算最优阈值（基于F-β）
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        f_beta_scores = [f_beta_score(p, r, beta) for p, r in zip(precision[:-1], recall[:-1])]
        optimal_idx = np.argmax(f_beta_scores)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_best = (y_pred_prob >= optimal_threshold).astype(int)

        # 保存指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred_best))
        metrics['auc'].append(roc_auc_score(y_test, y_pred_prob))
        metrics['precision'].append(precision_score(y_test, y_pred_best))
        metrics['recall'].append(recall_score(y_test, y_pred_best))
        metrics['f1'].append(f1_score(y_test, y_pred_best))
        metrics['optimal_threshold'].append(optimal_threshold)
        metrics['best_f_beta'].append(f_beta_scores[optimal_idx])

        current_auc = roc_auc_score(y_test, y_pred_prob)
        # 更新最佳折记录
        if current_auc > best_auc:
            best_auc = current_auc
            best_fold_data = {
                'fold_num': i,
                'y_test': y_test,
                'y_pred_prob': y_pred_prob,
                'model': model,
                'history': history
            }

        # 打印当前折结果
        print(f"\nFold {i} Metrics:")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"  AUC:      {metrics['auc'][-1]:.4f}")
        print(f"  Precision:{metrics['precision'][-1]:.4f}")
        print(f"  Recall:   {metrics['recall'][-1]:.4f}")
        print(f"  F1:       {metrics['f1'][-1]:.4f}")
        print(f"  F-{beta}:  {metrics['best_f_beta'][-1]:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print("\nClassification Report (Optimal Threshold):")
        print(classification_report(y_test, y_pred_best))

    # === 仅绘制最佳折的图表 ===
    if best_fold_data:
        # ROC曲线
        fpr, tpr, _ = roc_curve(best_fold_data['y_test'], best_fold_data['y_pred_prob'])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange',
                 label=f'Best Fold {best_fold_data["fold_num"]} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('MLP ROC Curve (Best Fold by AUC)')
        plt.legend(loc="lower right")
        plt.show()

        # # 训练曲线
        # history = best_fold_data['history'].history
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(history['loss'], label='Train Loss')
        # plt.plot(history['val_loss'], label='Val Loss')
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(history['accuracy'], label='Train Accuracy')
        # plt.plot(history['val_accuracy'], label='Val Accuracy')
        # plt.legend()
        # plt.suptitle(f'MLP Training History (Best Fold {best_fold_data["fold_num"]})')
        # plt.show()
        # 训练曲线
        history_dict = best_fold_data['history'].history
        plt.figure(figsize=(12, 4))

        # 第一张图：训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['loss'], label='Train Loss')
        plt.plot(history_dict['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training History - Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 第二张图：训练和验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['accuracy'], label='Train Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training History - Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'MLP Training History (Best Fold {best_fold_data["fold_num"]})')
        plt.show()

    # 打印汇总结果
    print("\n=== Cross-Validation Summary ===")
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1', 'best_f_beta']:
        print(f"{metric.capitalize():<10}: mean = {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")
    print(
        f"Optimal Threshold: mean = {np.mean(metrics['optimal_threshold']):.4f} ± {np.std(metrics['optimal_threshold']):.4f}")

    return metrics

def RandomForest_CAT(X_train, X_test, y_train, y_test):
    # 初始化随机森林
    rf_model = RandomForestClassifier(
        n_estimators=100,  # 树的数量（默认100）
        max_depth=None,  # 树的最大深度（不限制）
        min_samples_split=2,  # 分裂所需最小样本数
        random_state=42  # 固定随机种子
    )
    # 训练模型
    rf_model.fit(X_train, y_train)
    # 预测测试集
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]  # 正类的概率（用于AUC）
    # 打印评估报告
    print("分类报告：\n", classification_report(y_test, y_pred))
    print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))
    print("AUC得分：", roc_auc_score(y_test, y_proba))

    # 计算精确率-召回率曲线
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # 在0.4到0.5之间测试多个阈值
    thresholds = np.linspace(0.4, 0.5, 10)
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        print(f"阈值 {thresh:.2f}:")
        print(classification_report(y_test, y_pred))

    # # 找到使F1最大的阈值
    # f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    # best_threshold = thresholds[np.argmax(f1_scores)]
    # print("最佳阈值：", best_threshold)
    #
    # # 按新阈值重新预测
    # y_pred_optimized = (y_proba >= best_threshold).astype(int)
    # print(classification_report(y_test, y_pred_optimized))

def XGBoost_CAT(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # 二分类任务
        n_estimators=300,  # 树的数量
        max_depth=4,  # 控制树复杂度
        learning_rate=0.005,  # 学习率
        subsample=0.8,  # 行采样比例
        colsample_bytree=0.8,  # 列采样比例
        reg_alpha=0.1,  # L1正则
        reg_lambda=0.1,  # L2正则
        random_state=42,
        eval_metric=['auc','logloss'],  # 增加F1监控
        early_stopping_rounds=20
    )
    # 使用早停法训练
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10  # 打印训练日志
    )
    # 预测测试集
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 正类概率
    # 计算指标
    print("\nEvaluation Metrics:")
    print(f"- Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"- F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"- AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # 训练过程曲线
    results = model.evals_result()
    plt.figure(figsize=(10, 4))
    plt.plot(results['validation_0']['auc'], label='Train AUC')
    plt.plot(results['validation_1']['auc'], label='Test AUC')
    plt.axvline(model.best_iteration, color='gray', linestyle='--')
    plt.title('AUC during Training')
    plt.legend()
    plt.show()

def XGBoost_CAT_fold(folds, beta=1, epochs=500, batch_size=32):
    # 初始化存储每折指标的字典
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'optimal_threshold': [],
        'best_f_beta': []
    }
    best_auc = -1
    best_fold_data = None  # 保存最佳折的数据

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        # 进一步划分验证集（从训练集划分）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # 创建XGBoost模型
        model = xgb.XGBClassifier(
            n_estimators=epochs,  # 类似于MLP的epochs
            max_depth=6,  # 控制树的最大深度
            learning_rate=0.1,  # 学习率
            subsample=0.8,  # 类似于dropout
            colsample_bytree=0.8,  # 特征采样
            reg_alpha=0.1,  # L1正则化
            reg_lambda=0.1,  # L2正则化
            objective='binary:logistic',
            eval_metric=['logloss', 'auc', 'error'],
            early_stopping_rounds=20,  # 类似于early_stopping
            random_state=42
        )

        # 训练模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=0
        )

        # 预测概率
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算最优阈值（基于F-β）
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
        f_beta_scores = [f_beta_score(p, r, beta) for p, r in zip(precision[:-1], recall[:-1])]
        optimal_idx = np.argmax(f_beta_scores)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_best = (y_pred_prob >= optimal_threshold).astype(int)



        # 保存指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred_best))
        metrics['auc'].append(roc_auc_score(y_test, y_pred_prob))
        metrics['precision'].append(precision_score(y_test, y_pred_best))
        metrics['recall'].append(recall_score(y_test, y_pred_best))
        metrics['f1'].append(f1_score(y_test, y_pred_best))
        metrics['optimal_threshold'].append(optimal_threshold)
        metrics['best_f_beta'].append(f_beta_scores[optimal_idx])

        current_auc = roc_auc_score(y_test, y_pred_prob)
        # 更新最佳AUC记录
        if current_auc > best_auc:
            best_auc = current_auc
            best_fold_data = {
                'fold_num': i,
                'y_test': y_test,
                'y_pred_prob': y_pred_prob,
                'model': model,
                'history': model.evals_result()  # 保存训练历史
            }

        # 打印当前折结果
        print(f"\nFold {i} Metrics:")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"  AUC:      {metrics['auc'][-1]:.4f}")
        print(f"  Precision:{metrics['precision'][-1]:.4f}")
        print(f"  Recall:   {metrics['recall'][-1]:.4f}")
        print(f"  F1:       {metrics['f1'][-1]:.4f}")
        print(f"  F-{beta}:  {metrics['best_f_beta'][-1]:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print("\nClassification Report (Optimal Threshold):")
        print(classification_report(y_test, y_pred_best))
        print(f"实际训练轮次: {model.best_iteration}")

    # === 仅绘制最佳折的ROC曲线 ===
    if best_fold_data:
        fpr, tpr, _ = roc_curve(best_fold_data['y_test'], best_fold_data['y_pred_prob'])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange',
                 label=f'Best Fold {best_fold_data["fold_num"]} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('XGBoost ROC Curve (Best Fold by AUC)')
        plt.legend(loc="lower right")
        plt.show()

        # 绘制训练曲线
        history = best_fold_data['history']
        plt.figure(figsize=(12, 4))

        # 第一张图：训练和验证损失
        plt.subplot(1, 2, 1)
        plt.plot(history['validation_0']['logloss'], label='Train Loss')  # 注意：validation_0现在是训练集
        plt.plot(history['validation_1']['logloss'], label='Val Loss')  # validation_1是验证集
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Loss')
        plt.title('Training History - Loss')
        plt.legend()

        # 第二张图：训练和验证准确率
        plt.subplot(1, 2, 2)
        plt.plot([1 - x for x in history['validation_0']['error']], label='Train Accuracy')
        plt.plot([1 - x for x in history['validation_1']['error']], label='Val Accuracy')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Accuracy')
        plt.title('Training History - Accuracy')
        plt.legend()

        plt.suptitle(f'XGBoost Training History (Best Fold {best_fold_data["fold_num"]})')
        plt.show()

    # 打印汇总结果
    print("\n=== Cross-Validation Summary ===")
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1', 'best_f_beta']:
        print(f"{metric.capitalize():<10}: mean = {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")
    print(
        f"Optimal Threshold: mean = {np.mean(metrics['optimal_threshold']):.4f} ± {np.std(metrics['optimal_threshold']):.4f}")

    return metrics

def SVM_CAT(X_train, X_test, y_train, y_test):
    # 1.初始化SVM（RBF核）
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel='rbf',
            C=1.0,  # 正则化参数
            gamma='scale',  # 核函数带宽
            class_weight='balanced',  # 自动计算类别权重
            probability=True  # 启用概率预测（用于AUC）
        )
    )
    svm_model.fit(X_train, y_train)

    # 2. 预测概率和类别
    y_proba = svm_model.predict_proba(X_test)[:, 1]  # 正类概率
    y_pred = svm_model.predict(X_test)  # 默认阈值预测

    # 3. 计算PR曲线和最优阈值（基于F1）
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

    # 4. 计算所有评估指标
    metrics = {
        'default_threshold': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba)
        },
        'optimal_threshold': {
            'accuracy': accuracy_score(y_test, y_pred_optimal),
            'precision': precision_score(y_test, y_pred_optimal),
            'recall': recall_score(y_test, y_pred_optimal),
            'f1': f1_score(y_test, y_pred_optimal),
            'threshold': optimal_threshold
        },
        'auc_pr': auc(recall, precision)
    }

    # 5. 打印完整报告
    print("\n=== Default Threshold (0.5) ===")
    print(f"Accuracy:  {metrics['default_threshold']['accuracy']:.4f}")
    print(f"Precision: {metrics['default_threshold']['precision']:.4f}")
    print(f"Recall:    {metrics['default_threshold']['recall']:.4f}")
    print(f"F1 Score:  {metrics['default_threshold']['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['default_threshold']['auc_roc']:.4f}")

    print("\n=== Optimal Threshold (F1-maximizing) ===")
    print(f"Threshold: {metrics['optimal_threshold']['threshold']:.4f}")
    print(f"Accuracy:  {metrics['optimal_threshold']['accuracy']:.4f}")
    print(f"Precision: {metrics['optimal_threshold']['precision']:.4f}")
    print(f"Recall:    {metrics['optimal_threshold']['recall']:.4f}")
    print(f"F1 Score:  {metrics['optimal_threshold']['f1']:.4f}")
    print(f"AUC-PR:    {metrics['auc_pr']:.4f}")

    print("\nClassification Report (Optimal Threshold):")
    print(classification_report(y_test, y_pred_optimal))

    # 6. 可视化
    # 图1: ROC曲线
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'AUC-ROC = {metrics["default_threshold"]["auc_roc"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # 图2: PR曲线
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AUC-PR = {metrics["auc_pr"]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # 图3: 阈值优化
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--',
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.show()

    return svm_model, metrics

def SVM_CAT_fold(folds):
    # 初始化存储每折指标的字典
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'optimal_threshold': []
    }

    best_auc = -1
    best_fold_data = None  # 保存最佳折的数据

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        # 构建SVM模型（假设数据已标准化）
        model = make_pipeline(
            SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        )

        # 训练和预测
        model.fit(X_train, y_train)
        # 获取决策函数值（用于绘制训练过程）
        decision_values = model.decision_function(X_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # 正类概率

        # 计算PR曲线和最优阈值
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        #accuracies = [accuracy_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

        # 保存指标（使用最优阈值后的预测）
        metrics['accuracy'].append(accuracy_score(y_test, y_pred_optimal))
        metrics['auc'].append(roc_auc_score(y_test, y_proba))
        metrics['precision'].append(precision_score(y_test, y_pred_optimal))
        metrics['recall'].append(recall_score(y_test, y_pred_optimal))
        metrics['f1'].append(f1_score(y_test, y_pred_optimal))
        metrics['optimal_threshold'].append(optimal_threshold)

        # 计算当前AUC
        current_auc = roc_auc_score(y_test, y_proba)
        metrics['auc'].append(current_auc)

        # 更新最佳折记录
        if current_auc > best_auc:
            best_auc = current_auc
            best_fold_data = {
                'fold_num': i,
                'model': model,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'y_proba': y_proba,
                'decision_values': decision_values
            }


        # 打印当前折结果（与LR_CAT_CV格式一致）
        print(f"\nFold {i} Metrics:")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"  AUC:      {metrics['auc'][-1]:.4f}")
        print(f"  Precision:{metrics['precision'][-1]:.4f}")
        print(f"  Recall:   {metrics['recall'][-1]:.4f}")
        print(f"  F1:       {metrics['f1'][-1]:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print("\nClassification Report (Optimal Threshold):")
        print(classification_report(y_test, y_pred_optimal))

    # === 绘制最优折的图表 ===
    if best_fold_data:
        # 1. 绘制ROC曲线
        fpr, tpr, _ = roc_curve(best_fold_data['y_test'], best_fold_data['y_proba'])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange',
            label=f'Best Fold {best_fold_data["fold_num"]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SVM ROC Curve (Best Fold)')
        plt.legend(loc="lower right")
        plt.show()

        # 2. 绘制"训练过程" - 决策函数值分布
        plt.figure(figsize=(12, 5))

        # 决策函数值分布
        plt.subplot(1, 2, 1)
        sns.histplot(
            x=best_fold_data['decision_values'],
            hue=best_fold_data['y_train'],
            kde=True,
            bins=30,
            element="step"
            )
        plt.title('Decision Values Distribution')
        plt.xlabel('Decision Function Output')

        # 支持向量展示
        plt.subplot(1, 2, 2)
        sv_indices = np.abs(best_fold_data['model'].named_steps['svc'].decision_function(
            best_fold_data['X_train'])) <= 1
        plt.scatter(
            best_fold_data['X_train'][:, 0][sv_indices],
            best_fold_data['X_train'][:, 1][sv_indices],
            c='red', s=10, label='Support Vectors'
        )
        plt.scatter(
            best_fold_data['X_train'][:, 0],
            best_fold_data['X_train'][:, 1],
            c=best_fold_data['y_train'],
            cmap='coolwarm', alpha=0.3
            )
        plt.title('Support Vectors Visualization')
        plt.legend()

        plt.suptitle(f'SVM Training Analysis (Best Fold {best_fold_data["fold_num"]})')
        plt.tight_layout()
        plt.show()

    # 打印汇总结果（与LR_CAT_CV完全一致）
    print("\n=== Cross-Validation Summary ===")
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize():<10}: mean = {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")
    print(
        f"Optimal Threshold: mean = {np.mean(metrics['optimal_threshold']):.4f} ± {np.std(metrics['optimal_threshold']):.4f}")

    return metrics

def LR_CAT(X_train, X_test, y_train, y_test):
    # 构建逻辑回归模型（建议用Pipeline整合标准化）
    logreg_model = make_pipeline(
        StandardScaler(),  # 标准化（逻辑回归对特征尺度敏感）
        LogisticRegression(
            penalty='l2',  # L2正则化（默认）
            C=1.0,  # 正则化强度的倒数（越小正则化越强）
            solver='lbfgs',  # 适用于小到中型数据集
            max_iter=1000,  # 最大迭代次数
            class_weight='balanced',  # 自动平衡类别权重
            random_state=42  # 随机种子
        )
    )
    # 训练模型
    logreg_model.fit(X_train, y_train)

    # 预测与评估
    y_pred = logreg_model.predict(X_test)
    y_proba = logreg_model.predict_proba(X_test)[:, 1]  # 正类概率

    # 输出性能报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

def LR_CAT_fold(folds):
    # 初始化存储每折指标的字典
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    for i, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        # 构建模型（假设数据已标准化，Pipeline 中无需 StandardScaler）
        model = make_pipeline(
            LogisticRegression(
                penalty='l2', C=1.0, solver='lbfgs',
                max_iter=1000, class_weight='balanced', random_state=42
            )
        )
        # 训练和预测
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 计算所有指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['auc'].append(roc_auc_score(y_test, y_proba))
        metrics['precision'].append(precision_score(y_test, y_pred))
        metrics['recall'].append(recall_score(y_test, y_pred))
        metrics['f1'].append(f1_score(y_test, y_pred))

        # 打印当前折的详细报告
        print(f"\nFold {i} Metrics:")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"  AUC:      {metrics['auc'][-1]:.4f}")
        print(f"  Precision:{metrics['precision'][-1]:.4f}")
        print(f"  Recall:   {metrics['recall'][-1]:.4f}")
        print(f"  F1:       {metrics['f1'][-1]:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # 打印所有折的均值±标准差
    print("\n=== Overall Cross-Validation Results ===")
    for name, values in metrics.items():
        print(f"{name.capitalize():<10}: mean = {np.mean(values):.4f} ± {np.std(values):.4f}")

    return metrics


def LR_Keras_CAT_fold(folds, epochs=100, batch_size=32):
    metrics = {
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'optimal_threshold': []
    }

    best_auc = -1
    best_fold_history = None
    best_fold_data = None

    for i, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        # 构建Keras逻辑回归模型
        model = Sequential([
            Dense(1, input_dim=X_train.shape[1], activation='sigmoid',
                  kernel_regularizer=regularizers.l2(0.01))
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 训练模型（保留验证集用于早停）
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=20, restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                       ],
            verbose=0
        )
        #
        # 预测概率
        y_proba = model.predict(X_test).flatten()
        current_auc = roc_auc_score(y_test, y_proba)
        metrics['auc'].append(current_auc)

        # 更新最佳折记录
        if current_auc > best_auc:
            best_auc = current_auc
            best_fold_history = history.history
            best_fold_data = {
                'fold_num': i,
                'model': model,
                'X_test': X_test,
                'y_test': y_test,
                'y_proba': y_proba
            }

        # 计算最优阈值和指标
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_best = (y_proba >= optimal_threshold).astype(int)

        # 保存指标
        metrics['accuracy'].append(accuracy_score(y_test, y_pred_best))
        metrics['precision'].append(precision_score(y_test, y_pred_best))
        metrics['recall'].append(recall_score(y_test, y_pred_best))
        metrics['f1'].append(f1_score(y_test, y_pred_best))
        metrics['optimal_threshold'].append(optimal_threshold)

        # 打印当前折结果
        print(f"\nFold {i} Metrics:")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"  AUC:      {current_auc:.4f}")
        print(f"  Precision:{metrics['precision'][-1]:.4f}")
        print(f"  Recall:   {metrics['recall'][-1]:.4f}")
        print(f"  F1:       {metrics['f1'][-1]:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")

    # === 绘制最佳折的图表 ===
    if best_fold_data:
        # 1. ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(best_fold_data['y_test'], best_fold_data['y_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',
                 label=f'Fold {best_fold_data["fold_num"]} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LR ROC Curve (Best Fold)')
        plt.legend(loc="lower right")
        plt.grid(False)
        plt.show()
    if best_fold_history:
        plt.figure(figsize=(12, 4))  # 稍调高高度以适应标题

        # 左图：Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(best_fold_history['loss'],  linewidth=1.5, label='Train Loss')
        plt.plot(best_fold_history['val_loss'],  linewidth=1.5, label='Val Loss')

        # 标记最佳epoch
        best_epoch = np.argmin(best_fold_history['val_loss'])
        plt.axvline(best_epoch, color='k', linestyle=':', alpha=0.7,
                    label=f'Best Epoch ({best_epoch})')

        plt.xlabel('Epochs', fontsize=11)
        plt.ylabel('Binary Cross-Entropy', fontsize=11)
        plt.legend(framealpha=0.9, fontsize=10)
        plt.grid(False)

        # 右图：Accuracy曲线
        plt.subplot(1, 2, 2)
        plt.plot(best_fold_history['accuracy'],  linewidth=1.5, label='Train Accuracy')
        plt.plot(best_fold_history['val_accuracy'], linewidth=1.5, label='Val Accuracy')
        plt.axvline(best_epoch, color='k', linestyle=':', alpha=0.7)

        plt.xlabel('Epochs', fontsize=11)
        plt.ylabel('Accuracy', fontsize=11)
        plt.legend(framealpha=0.9, fontsize=10)
        plt.grid(False)

        # 主标题
        plt.suptitle(f'LR Training History (Best Fold {best_fold_data["fold_num"]})',
                     y=1, fontsize=12)
        plt.show()

        # 3. 显示最佳模型结构
        print("\nBest Model Summary:")
        best_fold_data['model'].summary()

    # 打印汇总结果
    print("\n=== Cross-Validation Summary ===")
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize():<10}: mean = {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")
    print(
        f"Optimal Threshold: mean = {np.mean(metrics['optimal_threshold']):.4f} ± {np.std(metrics['optimal_threshold']):.4f}")

    return metrics


if __name__ == '__main__':
    # 特征提取
    # results = process_all_eye_tracking_files("eye_data")
    # save_eye_tracking_metrics_to_csv(results,"feature.csv")

    # 拼接人口统计特征和眼动特征
    # merge_csv_files("factor.csv","feature.csv","result_withMMSE.csv")

    # 处理结果变量
    # processMMSE("feature_data/result_withMMSE.csv")

    # data = pd.read_csv('feature_data/result2_filter134_bal.csv')
    # 特征重要性分析
    # X, y = nomoralization(data)
    # feature_importance_RF(X, y)
    # feature_importance()

    # 训练数据

    train_data = pd.read_csv('feature_data/result_cat3_3_5_bal.csv')

    # X_train, X_test, y_train, y_test = feature_preprocess(train_data, True)

    # MLP_CAT(X_train, X_test, y_train, y_test)

    # RandomForest_CAT(X_train, X_test, y_train, y_test)

    # XGBoost_CAT(X_train, X_test, y_train, y_test)

    # SVM_CAT(X_train, X_test, y_train, y_test)

    # LR_CAT(X_train, X_test, y_train, y_test)

    ##################################################

    folds = feature_preprocess_fold(train_data,True)

    # LR_CAT_fold(folds)

    # SVM_CAT_fold(folds)

    # MLP_CAT_fold(folds)

    # XGBoost_CAT_fold(folds)

    LR_Keras_CAT_fold(folds)