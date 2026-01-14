import numpy as np
class VEBFNeuron:
    def __init__(self, center, label, n_dim, a_init):
        # 4.3 Initialize neuron parameters
        self.center = center
        self.label = label
        self.n_dim = n_dim
        self.n = 1  #จำนวนข้อมูลที่รับเข้ามา
    
        self.a = np.array(a_init)  #ค่า a เริ่มต้น
    
        self.cov_matrix = np.zeros((n_dim, n_dim))  #S เริ่มเป็น 0
    
        # เก็บ Eigenvalues/vectors ไว้ใช้ 
        self.eigenvalues = np.zeros(n_dim)
        self.eigenvectors = np.eye(n_dim)
    
    def update_eigen_properties(self):
        # 4.4 คำนวณ Eigenvalues และ Eigenvectors ของ Covariance Matrix
        if self.n > 1:
            cov_matrix_normalized = self.cov_matrix / (self.n - 1)
            self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_matrix_normalized)
        else:
            self.eigenvalues = np.zeros_like(self.eigenvalues)
            self.eigenvectors = np.eye(len(self.eigenvalues))
            
    def calculate_distance(self, x):
        # คำนวณระยะห่างระหว่างจุดข้อมูล x กับ center ของ neuron
        diff = x - self.center
        
        # หมุนแกนด้วย Eigenvectors (Projection onto U)
        projected = np.dot(diff, self.eigenvectors)
        normalized = projected / self.a 
        
        # ระยะทาง
        dist_sq = np.sum(normalized ** 2)
        return dist_sq
    
    def is_covering(self, x): #psi_cs
        # ตรวจสอบว่าจุดข้อมูล x อยู่ในขอบเขตของ neuron หรือไม่
        return self.calculate_distance(x)-1 <= 0