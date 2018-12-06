import os

import matplotlib.pyplot as plt
import numpy as np


class ODE(object):

    def __init__(self, f, x_0, y_0, x_n, num_node, num_iter=10):
        # dy/dx = f(x, y)
        # y_0 = y(x_0)
        # x_n：求解区间的右端点
        # num_node: 区间[x_0, x_n]内离散的节点数-1
        # num_iter: 隐式法迭代次数
        self.f = f
        self.y_0 = y_0
        self.x_0 = x_0
        self.x_n = x_n
        self.h = (x_n - x_0) / (num_node - 1)  # 步长
        self.x_nodes = np.linspace(x_0, x_n, num_node)
        self.num_node = num_node
        self.num_iter = num_iter

    def recur_explicit(self, phi):
        # 显式单步法递推 y_n+1 = y_n + h*phi(x_n, y_n)
        y = np.array([self.y_0])
        for x_i in self.x_nodes[:-1]:
            y_new = y[-1] + self.h * phi(x_i, y[-1])
            y = np.append(y, y_new)
        return y

    def recur_implicit(self, phi, num_iter):
        # 隐式单步法递推 y_n+1 = y_n + h*phi(x_n, x_n+1, y_n, y_n+1)
        # num_iter: 迭代次数
        y = np.array([self.y_0])
        for x_i in self.x_nodes[:-1]:
            y_new = y[-1] + self.h * self.f(x_i, y[-1])
            for n in np.arange(num_iter):
                y_new = y[-1] + self.h * phi(x_i, x_i+self.h, y[-1], y_new)
            y = np.append(y, y_new)
        return y

    def euler_forward(self):
        # 欧拉法
        phi = self.f
        y = self.recur_explicit(phi)
        return y

    def euler_backward(self):
        # 后退的欧拉法
        def phi(x_i, x_ip, y_i, y_ip):
            return self.f(x_ip, y_ip)
        y = self.recur_implicit(phi, num_iter=self.num_iter)
        return y

    def trap_method(self):
        # 梯形法
        def phi(x_i, x_ip, y_i, y_ip):
            return 0.5*(self.f(x_i, y_i) + self.f(x_ip, y_ip))
        y = self.recur_implicit(phi, num_iter=self.num_iter)
        return y

    def euler_improved(self):
        # 改进的欧拉法
        def phi(x_i, x_ip, y_i, y_ip):
            return 0.5*(self.f(x_i, y_i) + self.f(x_ip, y_ip))
        y = self.recur_implicit(phi, num_iter=1)
        return y

    def center_diff(self):
        # 中矩形法
        def phi(x_i, x_ip, y_i, y_ip):
            return self.f(0.5*(x_i+x_ip), 0.5*(y_i+y_ip))
        y = self.recur_implicit(phi, num_iter=self.num_iter)
        return y

    def runge_kutta_4(self):
        # 4阶龙格库塔方法
        def phi(x_i, y_i):
            h = self.h
            k_1 = self.f(x_i, y_i)
            k_2 = self.f(x_i + 0.5 * h, y_i + 0.5 * h * k_1)
            k_3 = self.f(x_i + 0.5 * h, y_i + 0.5 * h * k_2)
            k_4 = self.f(x_i + h, y_i + h * k_3)
            phi_value = 1/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            return phi_value
        y = self.recur_explicit(phi)
        return y

    def plot(self, y_anafunc, y_approx, y_approx_label, is_save=True):
        # y_anafunc: 解析解的数值函数
        # y_approx：数值解的函数值的集合
        # y_approx_label: 数值解的函数图像的标签
        # save: 是否保存图像文件，如果保存，将存于工作目录下的“plot_figure”
        # 将输出两张图，一张包含y_anafunc和y_approx，另一张包含残差
        x_ana = np.linspace(self.x_0, self.x_n, 100)
        y_ana = y_anafunc(x_ana)
        x_approx = np.linspace(self.x_0, self.x_n, y_approx.size)
        error_y = np.abs(y_approx - y_anafunc(x_approx))

        plt.figure()
        plt.plot(x_ana, y_ana, label='Analytic solution')
        plt.scatter(x_approx, y_approx,
                    color='r',
                    label=y_approx_label)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        if is_save is not False:
            dir_path = '.\\plot_figure'
            if os.path.exists(dir_path) is False:
                os.mkdir(dir_path)
            file_name = 'ODE_' + y_approx_label + '.jpg'
            file_path = os.path.join(dir_path, file_name)
            plt.savefig(file_path)

        plt.show()

        plt.figure()
        plt.plot(x_approx, error_y, label='Error_of_' + y_approx_label)
        plt.xlabel('x')
        plt.ylabel('Err')
        plt.legend()

        if is_save is not False:
            file_name = 'ODE_Err_of_' + y_approx_label + '.jpg'
            file_path = os.path.join('.\\plot_figure', file_name)
            plt.savefig(file_path)

        plt.show()


if __name__ == "__main__":
    # ========================= Config ==========================
    def f_(x, y):
        return y - 2 * x / y  # dy/dx = f(x, y)


    x_0_, y_0_ = 0, 1  # y(x_0) = y_0
    x_n_ = 1  # 数值解区间的右端点
    num_node_ = 10  # 数值求解区间的节点数


    def y_anafunc_(x): return (1 + 2 * x) ** 0.5  # 解析解


    # ===========================================================
    ode_test = ODE(f_, x_0_, y_0_, x_n_, num_node_)

    y_approx_ = ode_test.euler_forward()
    ode_test.plot(y_anafunc_, y_approx_, 'Euler_method')
    
    y_approx_ = ode_test.euler_backward()
    ode_test.plot(y_anafunc_, y_approx_, 'Implicit_Euler_method')

    y_approx_ = ode_test.trap_method()
    ode_test.plot(y_anafunc_, y_approx_, 'Trapezoid_method')

    y_approx_ = ode_test.center_diff()
    ode_test.plot(y_anafunc_, y_approx_, 'Center_difference_method')

    y_approx_ = ode_test.euler_improved()
    ode_test.plot(y_anafunc_, y_approx_, 'Improved_Euler_method')

    y_approx_ = ode_test.runge_kutta_4()
    ode_test.plot(y_anafunc_, y_approx_, 'RK4_method')
