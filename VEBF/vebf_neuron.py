import numpy as np
class VEBFNeuron:
    def __init__(self, center, label, n_dim, a_init, n_init = 1):
        # 4.3 Initialize neuron parameters
        self.center = center
        self.label = label
        self.n_dim = n_dim
        self.n = n_init
        
        self.a = np.array(a_init, dtype=float) #รัศมีแกนวงรี (Semi-axes lengths) 
    
        self.cov_matrix = np.zeros((n_dim, n_dim))  #S (Covariance Matrix) เริ่มเป็น 0
    
        # เก็บ Eigenvalues, vectors ไว้ใช้ 
        self.eigenvalues = np.zeros(n_dim)
        self.eigenvectors = np.eye(n_dim)
    
    def update_eigen_properties(self, cov_matrix=None):
        # คำนวณ Eigenvalues และ Eigenvectors ของ Covariance Matrix
        
        # ถ้าไม่ส่ง cov_matrix มาให้ใช้ของตัวเอง
        if cov_matrix is None:
            cov_matrix = self.cov_matrix
            
        # ข้อมูลน้อยเกินไป (สร้างวงรีไม่ได้) n < 2 ความแปรปรวนยังไม่เกิด หรือเป็น 0
        if self.n < 2 and cov_matrix is self.cov_matrix:
            self.eigenvalues = np.zeros_like(self.eigenvalues)
            self.eigenvectors = np.eye(len(self.eigenvalues))
            return self.eigenvalues, self.eigenvectors
            
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)
        self.eigenvalues = self.eigenvalues[::-1] # เรียงจากมากไปน้อย เพราะ np.linalg.eigh คืนค่าจากน้อยไปมาก
        self.eigenvectors = self.eigenvectors[:, ::-1]
        
        self.eigenvalues = np.maximum(self.eigenvalues, 0)  # ป้องกัน eigenvalues ติดลบจากความคลาดเคลื่อนทางคณิตศาสตร์
        return self.eigenvalues, self.eigenvectors
    
    def recursive_mean(self, x_new):
        N = self.n
        center_new = (N * self.center + x_new) / (N + 1)
        return center_new

    def recursive_covariance(self, x_new, center_new):
        N = self.n
        term1 = N / (N + 1) * self.cov_matrix 
        diff = (center_new - x_new).reshape(-1, 1) # column vector
        term2 = (1 / N)* np.dot(diff, diff.T) if N > 0 else np.zeros_like(self.cov_matrix)
        cov_new = term1 + term2
        return cov_new
            
    def calculate_psi(self, x, center=None, eigenvectors=None, a=None):
        # สมการ 4.33 psi = Sum( ((x - c)^T * u_i)^2 / a_i^2 ) - 1
        
        # ถ้าไม่ส่งค่า center, eigenvectors, a มา ให้ใช้ของตัวเอง (เช็คทั่วไป)
        if center is None:
            center = self.center
        if eigenvectors is None:
            eigenvectors = self.eigenvectors
        if a is None:
            a = self.a
        
        # 1. คำนวณระยะห่างระหว่างจุดข้อมูล x กับ center ของ neuron
        diff = x - center
        
        # 2. Projection บน Eigen (หมุนแกนด้วย Eigenvectors)
        projected = np.dot(diff, eigenvectors)
        try:
            normalized = projected / a
        except ZeroDivisionError:
            print("Error: Division by zero in calculate_psi, a is 0")
            exit(1)
        
        # 3. คำนวณค่า psi
        psi = np.sum(normalized ** 2) -1
        return psi
    
    def calculate_provisional_params(self, x_new):
        # b. คำนวณหา จุดศูนย์กลางใหม่ , Covariance matrix ใหม่ และeigenvector ใหม่ ที่ทำให้คลุมมากขึ้น ยังไม่ปรับขนาดแกนนะ ใช้ $a_{old}$
        # คำนวณค่าใหม่ชั่วคราวเพื่อเช็คว่าจะเขยิบไปรับหรือสร้างNueraonใหม่
        
        x_new = np.array(x_new)
        N = self.n
        center_new = self.recursive_mean(x_new)
        cov_new = self.recursive_covariance(x_new, center_new)
        
        return center_new, cov_new
    
    def update_params(self, x_new, center_new, cov_new, n0=2):
        # c. ถ้า psi <= 0 อัพเดต
        center_old = self.center.copy()
        
        self.center = center_new
        self.cov_matrix = cov_new
        self.n += 1
        
        self.update_eigen_properties()
        
        # ถ้าขยับแล้วคลุมข้อมูลเก่าไม่หมด เพิ่ม a
        if self.n > n0:
            shift_vecter = self.center - center_old
            projected_shift = np.abs(np.dot(shift_vecter, self.eigenvectors))
            self.a += projected_shift
            
    
    def is_covering(self, x): #psi_cs
        # ตรวจสอบว่าจุดข้อมูล x อยู่ในขอบเขตของ neuron หรือไม่
        return self.calculate_psi(x)-1 <= 0
    
    def calculate_merge_psi(self, other_neuron):
        # psi สำหรับ merge neuron ตัวนี้ กับ other_neuron
        center_diff = self.center - other_neuron.center
        projected = np.dot(center_diff, other_neuron.eigenvectors)
        
        try:
            normalized = projected / other_neuron.a
        except ZeroDivisionError:
            print("Error: Division by zero in calculate_psi, other nueron a is 0")
            exit(1)
        
        psi_merge = np.sum(normalized ** 2) -1
        return psi_merge
      
    def merge_with(self, other_neuron):
        #รวมร่าง neuron นี้กับอีกอัน เสร็จแล้วลบอีกนั้นทิ้ง เก็บในตัวนี้
        N1 = self.n
        N2 = other_neuron.n
        N_merge = N1 + N2
        
        center_merge = (N1 * self.center + N2 * other_neuron.center) / N_merge
        
        # cov_merge
        term1 = (N1 * self.cov_matrix + N2 * other_neuron.cov_matrix) / N_merge
        diff = (self.center - other_neuron.center).reshape(-1, 1) 
        term2 = (N1 * N2 / (N_merge **2))*np.dot(diff, diff.T)
        cov_merge = term1 + term2
        
        # อัพเดต merge เข้าตัวนี้ self
        self.center = center_merge
        self.cov_matrix = cov_merge
        self.n = N_merge
        
        # หา a แกนวงรีใหม่  = \sqrt{2\pi |\lambda_i|}$  ; eigenvalues คำนวณใหม่หลังรวม
        eigenvalues_merge, eigenvectors_merge =  self.update_eigen_properties()
        a_merge = np.sqrt(2 * np.pi * np.abs(eigenvalues_merge))