import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vebf_network import VEBFNetwork

def main():
    # 1. โหลดข้อมูล: Breast Cancer (30 Features)
    print("Loading Breast Cancer Dataset (30 Dimensions)...")
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Normalization (สำคัญมากสำหรับ High Dimension)
    # ข้อมูลแต่ละช่องหน่วยต่างกัน (เช่น พื้นที่ vs ความเรียบ) ต้องปรับให้สเกลเดียวกันก่อน
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. แบ่งข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. เทรน VEBF (เทรนบน 30 มิติของจริง!)
    print(f"\nTraining on {X_train.shape[1]} dimensions...")
    # ข้อมูลเยอะและมิติเยอะ อาจต้องปรับ theta ให้ merge ง่ายขึ้นหน่อย ไม่งั้น neuron จะล้น
    net = VEBFNetwork(X_train=X_train, delta=2.0, n0=5, theta=0.9) 
    net.train(X_train, y_train)
    
    print(f"Done! Created {len(net.neurons)} neurons.")

    # 4. วัดผล
    y_pred = net.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> Test Accuracy: {acc * 100:.2f}% <<<")

    # 5. Visualization: ใช้ PCA ยุบ 30 มิติ เหลือ 2 มิติ เพื่อวาดกราฟ
    print("\nVisualizing with PCA (2D Projection)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test) # ยุบข้อมูล Test

    # วาดจุดข้อมูล (แยกสีตาม Class จริง)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm', alpha=0.6, edgecolors='k')
    
    # วาดจุดที่ทายผิด (กากบาทสีดำ)
    incorrect_idx = np.where(y_pred != y_test)[0]
    if len(incorrect_idx) > 0:
        plt.scatter(X_pca[incorrect_idx, 0], X_pca[incorrect_idx, 1], 
                    marker='x', c='black', s=100, linewidth=2, label='Prediction Error')

    plt.title(f'VEBF Prediction on Breast Cancer (PCA Projection)\nAccuracy: {acc*100:.2f}%')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nNote: กราฟนี้คือการยุบ 30 มิติเหลือ 2 มิติเพื่อการมองเห็น")
    print("เราจะไม่เห็นวงรี (Ellipsoids) เพราะวงรีอยู่ในโลก 30 มิติ")
    print("แต่เราจะเห็นว่า VEBF แยกแยะคลาสข้อมูลใน High-Dim ได้ดีแค่ไหน")

if __name__ == "__main__":
    main()