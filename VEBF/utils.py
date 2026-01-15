import numpy as np

def calculate_initial_a(X, delta=0.1, max_samples=1000):
    # 5.1 คำนวณค่า a เริ่มต้น จาก 
    # X : ข้อมูล training ทั้งหมด ; X.shape = (N, n_dim) แถว=จำนวนข้อมูล คอลัมน์=จำนวนฟีเจอร์
    # delta : ตัวคูณปรับค่า
    # max_samples : จำนวนตัวอย่างที่สุ่มมาใช้คำนวณ
    
    N = X.shape[0] # จำนวนข้อมูล
    n_dim = X.shape[1] # จำนวน features ของข้อมูล
    
    # 1.ถ้าจำนวนข้อมูลมากเกินไป สุ่มแค่บางข้อมูลมาคำนวณ
    if N > max_samples:
        # สุ่มเลขแถวของข้อมูลที่จะเอามาใช้คำนวณ เก็บไว้ใน array indices
        indices = np.random.choice(N, size=max_samples, replace=False) # replace=False คือสุ่มแบบหยิบออกมาไม่ใส่คืน
        data_to_calc = X[indices]
        curr_N = max_samples
    else:
        data_to_calc = X
        curr_N = N
    
    # 2.คำนวณระยะห่างระหว่างจุดข้อมูลแต่ละคู่
    # ใช้ Broadcasting ของ Numpy จะเร็วกว่าวน Loop
        # newaxis คือทำเมตริกให้มีมิติใหม่เพิ่มขึ้นมาเพื่อให้สามารถลบกันได้
    diff = data_to_calc[:, np.newaxis, :] - data_to_calc[np.newaxis, :, :] #แนวตั้ง - แนวนอน ได้ตาราง N*N ระยะห่างของจุด(แถว)-จุด(หลัก) ดังนั้นแนวทแยงจะเป็น ห่างตัวเอง = 0 สามเหลี่ยมบนล่างที่แบ่งด้วยแนวทแยงเท่ากัน พอเอามา sumทุกตัวในเมตริกนี้ ก็จะได้
    
    #หา d ใช้ Euclidean distance d= sqrt(sum((x_i - x_j)^2))
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # 3. หา d_av (Average Distance)
    # สูตร: (1/N^2) * sum(dists)
    d_av = np.sum(dists) / (curr_N ** 2)
    
    # 4. คืนค่า a vector (ขนาดเท่า dimension ของข้อมูล)
    # เปเปอร์บอกให้ใช้ค่าเท่ากันทุกแกนในตอนแรก
    return np.full(n_dim, delta * d_av)

def calculate_initial_width_hybrid(X, n_classes, max_samples=1000):
    """
    Algorithm 1: Initial Width Vector Algorithm for Hybrid LDA-PCA
    """
    N = X.shape[0]
    
    # 1. Sampling if too large
    if N > max_samples:
        indices = np.random.choice(N, size=max_samples, replace=False)
        data = X[indices]
    else:
        data = X
        
    # 2. Calculate pairwise distances
    # diff shape: (N, N, n_dim)
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    
    # dists shape: (N, N)
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Get upper triangle values (excluding diagonal 0)
    # distance values i != j
    dist_values = dists[np.triu_indices(data.shape[0], k=1)]
    
    if len(dist_values) == 0:
        return np.ones(X.shape[1]) # Fallback

    # Sort D
    D = np.sort(dist_values)
    min_D = D[0]
    max_D = D[-1]
    
    # 3. Intervals
    n_intervals = n_classes
    if n_intervals < 1: n_intervals = 1
    
    gamma = (max_D - min_D) / n_intervals
    
    if gamma == 0:
        return np.full(X.shape[1], max_D if max_D > 0 else 1.0)

    # 4. Count pairs in intervals
    # We can use histogram
    # bins edges: min_D, min_D + gamma, ..., max_D
    counts, bin_edges = np.histogram(D, bins=n_intervals, range=(min_D, max_D))
    
    # 5. Select category with most data vectors
    # argmax gives the index of the interval
    best_idx = np.argmax(counts)
    
    # The interval starts at bin_edges[best_idx]
    r = bin_edges[best_idx]
    
    # 6. Calculate omega
    # omega = r + gamma / 2
    omega = r + (gamma / 2)
    
    n_dim = X.shape[1]
    return np.full(n_dim, omega)


