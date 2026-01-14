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