from vebf_neuron import VebfNeuron
from utils import calculate_initial_a

class VebfNetwork:
    def __init__(self, X_train, delta=1.0, n0=5, theta=-0.5):
        # X_train : ข้อมูล training ทั้งหมด
        # delta : ตัวคูณปรับค่า a0
        #  คำนวณ a0 ตามสูตร 5.1
        self.a0 = calculate_initial_a(X_train, delta)
        print(f"calculated initial a0: {self.a0}")
        
        self.n0 = n0  # จำนวนตัวอย่างที่ใช้ในการอัพเดท a
        self.theta = theta  # ค่า threshold ในการเพิ่ม neuron ใหม่
        self.neurons = []  # array เก็บ neuron ทั้งหมดใน network
    
    def create_neuron(self, x, t):
        # สร้าง neuron ใหม่ ใช้ self.a0 เป็นค่า a เริ่มต้น
        neuron = VebfNeuron(center = x, label = t, n_dim = len(x), a_init=self.a0.copy()) # ใช้ self.a0.copy() เพื่อไม่ให้ Neuron แก้ค่าต้นฉบับ
        self.neurons.append(neuron)
        return neuron