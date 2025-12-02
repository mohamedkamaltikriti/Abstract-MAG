import time
# 💥 الاستيراد الحاسم: استدعاء الدوال من الملف الجديد
from tsp_functions import tsp_cost, tsp_neighbor, tsp_kick

# ==================== Abstract MAG Core ====================
class AbstractMAG:
    """
    Abstract MAG Core Framework - Optimized for high-scaling problems.
    Uses imported TSP functions for cost, neighbor finding, and kicking.
    """
    def __init__(self, problem_data, alpha=2, kick_every=15, time_limit=30.0):
        # تم تعيين الدوال الآن عبر الاستيراد بدلاً من المدخلات
        self.obj_func = tsp_cost 
        self.neighbor = tsp_neighbor
        self.kick = tsp_kick

        self.data = problem_data
        self.alpha = alpha
        self.kick_every = kick_every
        self.time_limit = time_limit

    def solve(self, initial_solution):
        path = initial_solution.copy()
        best_path = path.copy()
        best_cost = self.obj_func(path, self.data)

        start = time.time()
        kick_count = 0

        while time.time() - start < self.time_limit:
            improved = True
            while improved and time.time() - start < self.time_limit:
                path, cost, improved = self.neighbor(path, self.data, self.alpha)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()

            kick_count += 1
            if kick_count >= self.kick_every:
                path = self.kick(best_path, self.data)
                kick_count = 0

        return best_path, best_cost
