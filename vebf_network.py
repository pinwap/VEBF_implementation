import numpy as np
from vebf_neuron import VEBFNeuron
from utils import calculate_initial_a

class VEBFNetwork:
    def __init__(self, X_train, delta=1.0, n0=5, theta=-0.5):
        # X_train : ข้อมูล training ทั้งหมด
        # delta : ตัวคูณปรับค่า a0
        # n0 : จำนวนตัวอย่างที่ใช้ในการอัพเดท a
        # theta : ค่า threshold ในการ merge
        
        #  คำนวณ a0 ตามสูตร 5.1
        self.a0 = calculate_initial_a(X_train, delta)
        print(f"calculated initial a0: {self.a0}")
        
        self.n0 = n0  
        self.theta = theta  
        self.neurons = []  # array เก็บ neuron ทั้งหมดใน network
    
    def train(self, X_train, y_train):
        # VEBF Learning Algorithm (One-Pass)
        # วน Loop ข้อมูลทีละตัว
        for i, (x, t) in enumerate(zip(X_train, y_train)):
            #print(f"Training sample {i+1}/{len(X_train)}")
            self.learn_one_pass(x,t)
            
            # ดูความคืบหน้าทุก 50 ตัวอย่าง
            if (i+1) % 50 == 0:
                print(f"Trained on {i+1}/{len(X_train)} samples, current number of neurons: {len(self.neurons)}")
            
    def learn_one_pass(self, x, t):
        # 1. หา neuron ที่ class ตรงกัน และใกล้สุด
        closed_neuron = None
        min_psi = float('inf')
        
        candidate_neurons = [neuron for neuron in self.neurons if neuron.label == t]
        for neuron in candidate_neurons:
            psi = np.linalg.norm(x - neuron.center)
            if psi < min_psi:
                min_psi = psi
                closed_neuron = neuron
        
        # ถ้าไม่มี -> สร้างใหม่
        if closed_neuron is None:
            self.create_neuron(x, t)
            return
        
        # ถ้ามี -> ลองว่าครอบได้ไหม
        center_new, cov_new = closed_neuron.calculate_provisional_params(x)
        eigenvalues_new, eigenvectors_new = np.linalg.eigh(cov_new)
        eigenvalues_new = eigenvalues_new[::-1] # เรียงจากมากไปน้อย เพราะ np.linalg.eigh คืนค่าจากน้อยไปมาก
        
        # ดูว่าย้ายไป center ใหม่แล้วครอบไหม ถ้าใช้ a เดิม
        psi_new = closed_neuron.calculate_psi(x, center=center_new, eigenvectors=eigenvectors_new)
        
        if psi_new <= 0:
            # ครอบได้ -> อัพเดตพารามิเตอร์
            closed_neuron.update_params(x, center_new, cov_new, n0=self.n0)  
            
            # ลอง merge กับ neuron ใกล้ๆ อีก
            self.attempt_merge(closed_neuron)  
        
        else:
            # ครอบไม่ได้ -> สร้าง neuron ใหม่
            self.create_neuron(x, t)
    
    def create_neuron(self, x, t):
        # สร้าง neuron ใหม่ ใช้ self.a0 เป็นค่า a เริ่มต้น
        neuron = VEBFNeuron(center = x, label = t, n_dim = len(x), a_init=self.a0.copy()) # ใช้ self.a0.copy() เพื่อไม่ให้ Neuron แก้ค่าต้นฉบับ
        self.neurons.append(neuron)
    
    def attempt_merge(self, target_neuron):
        # ลูปเช็ค neuron ตัวอื่นที่มี label เดียวกัน
        for other_neuron in self.neurons[:]: # [:] เพื่อ copy list เวลาลบจะได้ไม่พัง
            if other_neuron is target_neuron or other_neuron.label != target_neuron.label:
                continue #ข้ามไปลูปถัดไป
            
            # คำนวณ psi_merge
            psi_merge1 = target_neuron.calculate_merge_psi(other_neuron)
            psi_merge2 = other_neuron.calculate_merge_psi(target_neuron)
            
            if psi_merge1 <= self.theta or psi_merge2 <= self.theta:
                # merge other_neuron เข้า target_neuron
                target_neuron.merge_with(other_neuron)
                
                # ลบ other_neuron ออกจาก network
                if other_neuron in self.neurons:
                    self.neurons.remove(other_neuron)
                break
            
    def predict(self, X_test):
        # หา neuron ที่ให้ค่า psi ต่ำสุด (ใกล้สุด)
        prediction = []
        for x in X_test:
            best_neuron = None
            min_psi = float('inf')
            
            for neuron in self.neurons:
                psi = neuron.calculate_psi(x)
                if psi < min_psi:
                    min_psi = psi
                    best_neuron = neuron
            
            if best_neuron:
                prediction.append(best_neuron.label)
            else:
                prediction.append(-1)  # กรณีไม่มี neuron เลย
                
        return np.array(prediction)