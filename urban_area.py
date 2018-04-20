"""
    Urban area interpretator.
    Author: Volodymyr Pavliukevych.
"""
from random import randint

class Point(object):
    "Point in coordinate system."
    def __init__(self, x=0, y=0):
        "Init attribute"
        super().__init__()
        self.point_x = x
        self.point_y = y
    def __repr__(self):
        return "Point(x: " + str(self.point_x) + ", y: " + str(self.point_y) + ")"
    def __str__(self):
        return "Point(x: " + str(self.point_x) + ", y: " + str(self.point_y) + ")"

    @classmethod
    def random(cls, min_x=0, max_x=0, min_y=0, max_y=0):
        "Return random coordinate"
        return Point(x=randint(min_x, max_x), y=randint(min_y, max_y))

    def get_x(self):
        "Return X component"
        return self.point_x
    def get_y(self):
        "Return Y component"
        return self.point_y
    def step_left(self):
        "Move left"
        self.point_x -= 1

    def step_right(self):
        "Move right"
        self.point_x += 1

    def step_up(self):
        "Move up"
        self.point_y -= 1

    def step_down(self):
        "Move down"
        self.point_y += 1


class UrbanArea(object):
    "Simple clock encoder"
    rows = 10
    columns = 10

    vacab_pad_key = 0
    vacab_unk_pad = 1
    vacab_go_key = 2
    vacab_eos_key = 3
    vacab_spare_key = 4

    action_up_key = 5
    action_down_key = 6
    action_left_key = 7
    action_right_key = 8
    action_take_key = 9
    action_attack_key = 10

    area_space_key = 11
    area_left_top_key = 12
    area_right_top_key = 13
    area_left_bottom_key = 14
    area_right_bottom_key = 15
    area_left_key = 16
    area_right_key = 17
    area_top_key = 18
    area_bottom_key = 19
    area_source_key = 20
    area_enemy_key = 21
    area_destenation_key = 22
    area_food_key = 23

    chars = ["<PAD>", "<UNK>", "<GO>", "<EOS>", "<SPARE>", "↑", "↓", "←", "→", "⤴", "🔥", "◯", "⬂", "⬃", "⬀", "⬁", "▶", "◀", "▼", "▲", "⦿", "🕷", "⊛", "🍎", ]

    @classmethod
    def area_length(cls):
        "Return length of input vector"
        return cls.rows * cls.columns

    @classmethod
    def cell_number_from_point(cls, point):
        "Return cell number"
        return (point.get_x() * UrbanArea.columns) + point.get_y()

    def __init__(self):
        "Init attribute"
        super().__init__()
        self.max_x_index = UrbanArea.columns - 1
        self.max_y_index = UrbanArea.rows - 1

        self.area = [[self.area_space_key] * UrbanArea.columns for n in range(UrbanArea.rows)]
        # sett walls
        for x_point in range(0, self.columns):
            if x_point == 0:
                self[x_point, 0] = UrbanArea.area_left_top_key
                self[x_point, self.max_y_index] = UrbanArea.area_left_bottom_key
            elif x_point == self.max_x_index:
                self[x_point, 0] = UrbanArea.area_right_top_key
                self[x_point, self.max_y_index] = UrbanArea.area_right_bottom_key
            else:
                self[x_point, 0] = UrbanArea.area_top_key
                self[x_point, self.max_y_index] = UrbanArea.area_bottom_key
        for y_point in range(1, self.max_y_index):
            self[0, y_point] = UrbanArea.area_left_key
            self[self.max_x_index, y_point] = UrbanArea.area_right_key
        # Set points
        self.source_point = Point.random(max_x=self.max_x_index, max_y=self.max_y_index)
        for _ in range(0, 10):
            food = Point.random(max_x=self.max_x_index, max_y=self.max_y_index)
            self.set_valule_at(food, UrbanArea.area_food_key)
        for _ in range(0, 10):
            enemy = Point.random(max_x=self.max_x_index, max_y=self.max_y_index)
            self.set_valule_at(enemy, UrbanArea.area_enemy_key)

        self.destination_point = Point.random(max_x=self.max_x_index, max_y=self.max_y_index)
        self[self.source_point.get_x(), self.source_point.get_y()] = UrbanArea.area_source_key
        self[self.destination_point.get_x(), self.destination_point.get_y()] = UrbanArea.area_destenation_key

    def set_valule_at(self, point, value):
        "Set value for area by point"
        self[point.get_x(), point.get_y()] = value

    def __setitem__(self, index, value):
        "Set area by points"
        self.area[index[0]][index[1]] = value

    def __getitem__(self, index):
        "Get area by points"
        return self.area[index[0]][index[1]]

    def get_area(self):
        "Getter for area"
        return self.area

    def output(self):
        "Return flatten area"
        return [item for sublist in self.area for item in sublist]

    def get_road(self):
        "Return right road from source to destination point."
        steps = []
        current_point = self.source_point
        def take_or_attack():
            "Add take action if there is food."
            if self[current_point.get_x(), current_point.get_y()] == UrbanArea.area_food_key:
                steps.append(UrbanArea.action_take_key)
            elif self[current_point.get_x(), current_point.get_y()] == UrbanArea.area_enemy_key:
                steps.append(UrbanArea.action_attack_key)

        while current_point.get_x() < self.destination_point.get_x():
            current_point.step_right()
            steps.append(UrbanArea.action_right_key)
            take_or_attack()
        while current_point.get_x() > self.destination_point.get_x():
            current_point.step_left()
            steps.append(UrbanArea.action_left_key)
            take_or_attack()
        while current_point.get_y() > self.destination_point.get_y():
            current_point.step_up()
            steps.append(UrbanArea.action_up_key)
            take_or_attack()
        while current_point.get_y() < self.destination_point.get_y():
            current_point.step_down()
            steps.append(UrbanArea.action_down_key)
            take_or_attack()
        steps.append(UrbanArea.vacab_eos_key)
        return steps
    def debug(self):
        "Draw area as text image."
        for row in range(0, UrbanArea.rows):
            for column in range(0, UrbanArea.columns):
                char = self[column, row]
                print(self.chars[char], end=' ')
            print("")
        for step in self.get_road():
            print(self.chars[step], end=' ')
        print("")
        print("source: ", self.chars[UrbanArea.area_source_key])
        print("destenation: ", self.chars[UrbanArea.area_destenation_key])
        print("food", self.chars[UrbanArea.area_food_key])
        print("enemy", self.chars[UrbanArea.area_enemy_key])
