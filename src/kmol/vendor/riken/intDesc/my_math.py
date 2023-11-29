"""計算処理定義モジュール"""
import math

import numpy as np


def distance_two_points(point_a, point_b):
    """2点間距離

    Args:
        point_a (list): 点Aの3次元座標
        point_b (list): 点Bの3次元座標

    Returns:
        float: 2点間距離
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    distance = np.linalg.norm(point_a - point_b)
    return distance


def vector_two_points(point_a, point_b):
    """2点がなすベクトル

    Args:
        point_a (list): 点Aの3次元座標
        point_b (list): 点Bの3次元座標

    Returns:
        list: 2点がなすベクトル
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    vector = point_b - point_a
    return vector.tolist()


def angle_two_vectors(vector_a, vector_b):
    """2つのベクトルのなす角

    Args:
        vector_a (list): ベクトルAの成分
        vector_b (list): ベクトルBの成分

    Returns:
        float: 2つのベクトルのなす角 [度]
    """
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    dot = np.dot(vector_a, vector_b)
    magnitude = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    rad = np.arccos(dot / magnitude)
    deg = math.degrees(rad)
    return deg


def angle_three_points(point_a, point_b, point_c):
    """3点のなす角

    Args:
        point_a (list): 点Aの3次元座標
        point_b (list): 点Bの3次元座標
        point_c (list): 点Cの3次元座標

    Returns:
        float: 3点のなす角 [度]
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    return angle_two_vectors(vector_ba, vector_bc)


def normal_vector_three_points(point_a, point_b, point_c):
    """3点のなす法線ベクトル

    Args:
        point_a (list): 点Aの3次元座標
        point_b (list): 点Bの3次元座標
        point_c (list): 点Cの3次元座標

    Returns:
        list: 3点のなす法線ベクトル
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)

    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
    cross = np.cross(vector_ba, vector_bc)
    return cross.tolist()


def distance_point_to_plane(point_a, point_b, point_c, point_p):
    """点と平面の距離

    Args:
        point_a (list): 点Aの3次元座標(平面を構成する点)
        point_b (list): 点Bの3次元座標(平面を構成する点)
        point_c (list): 点Cの3次元座標(平面を構成する点)
        point_p (list): 点Pの3次元座標(平面との距離を測る点)

    Returns:
        float: 点と平面の距離
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_p = np.array(point_p)

    normal_vector = normal_vector_three_points(point_a, point_b, point_c)
    unit_vector = normal_vector / np.linalg.norm(normal_vector)
    distance = np.linalg.norm(np.dot(unit_vector, point_p) - np.dot(unit_vector, point_a))

    return distance


def intersection_point_vertical_line_and_plane(point_a, point_b, point_c, point_p):
    """点から下ろした垂線と平面との交点

    Args:
        point_a (list): 点Aの3次元座標(平面を構成する点)
        point_b (list): 点Bの3次元座標(平面を構成する点)
        point_c (list): 点Cの3次元座標(平面を構成する点)
        point_p (list): 点Pの3次元座標(平面に対して垂線を下す点)

    Returns:
        list: 点から下ろした垂線と平面との交点
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_p = np.array(point_p)

    normal_vector = normal_vector_three_points(point_a, point_b, point_c)
    unit_vector = normal_vector / np.linalg.norm(normal_vector)

    vector_pa = point_a - point_p
    vector_oh = point_p + np.dot(unit_vector, vector_pa) * unit_vector

    return vector_oh.tolist()


def intersection_point_vertical_line_and_line(vector_v, point_a, point_p):
    """点から下した垂線とベクトルで定義される直線との交点

    Args:
        vector_v (list): ベクトルvの成分
        point_a (list): ベクトルvがなす直線上に存在する点Aの3次元座標
        point_p (list): 点Pの3次元座標

    Returns:
        list: 点から下ろした垂線と平面との交点
    """
    vector_v = np.array(vector_v)
    assert np.linalg.norm(vector_v) != 0.0, "Vector v magnitude is 0"
    point_a = np.array(point_a)
    point_p = np.array(point_p)

    vector_ap = point_p - point_a
    unit_vector = vector_v / np.linalg.norm(vector_v)
    vector_ab = np.dot(unit_vector, vector_ap) * unit_vector
    vector_ob = point_a + vector_ab

    return vector_ob.tolist()


def dihedral_angle_four_points(point_a, point_b, point_c, point_d):
    """4点がなす二面角

    Args:
        point_a (list): 点Aの3次元座標
        point_b (list): 点Bの3次元座標
        point_c (list): 点Cの3次元座標
        point_d (list): 点Dの3次元座標

    Returns:
        float: 4点がなす二面角 [度]
    """
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
    point_d = np.array(point_d)

    normal_vector_b = normal_vector_three_points(point_a, point_b, point_c)
    normal_vector_c = normal_vector_three_points(point_b, point_c, point_d)

    return angle_two_vectors(normal_vector_b, normal_vector_c)


def center_of_gravity(points):
    """N点間の重心

    Args:
        points (list): 要素数「N×3」の二次元配列

    Returns:
        list: N点間の重心
    """
    points = np.array(points, dtype=object)
    mean = np.mean(points, axis=0, keepdims=True)[0]
    return mean.tolist()
