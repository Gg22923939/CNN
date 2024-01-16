# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:19:44 2023

@author: user
"""

import Augmentor
p = Augmentor.Pipeline(r"C:\Users\user\Desktop\Python_Script\CNN\Eardrum_four")

p.rotate90(probability=1)
p.rotate270(probability=1)
p.process()
p.remove_operation(0)
p.remove_operation(0)

p.rotate90(probability=1)
p.process()
p.remove_operation(0)

p.rotate180(probability=1)
p.process()
p.remove_operation(0)

p.rotate270(probability=1)
p.process()
p.remove_operation(0)


p.flip_left_right(probability=1)
p.process()
p.remove_operation(0)


p.flip_top_bottom(probability=1)
p.process()
p.remove_operation(0)


p.status()
