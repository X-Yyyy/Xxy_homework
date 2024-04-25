#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/21 15:21
# @Author : Xxy.

import os,csv
if __name__ == '__main__':
    with open("./csv_testset", 'r') as csv_in:
        reader = csv.reader(csv_in)
        list_of_rows_with_first_frame = [row for row in reader if row[4] == "L_raw" and row[6] == 'True']
        print(list_of_rows_with_first_frame)