import unittest
import subprocess
import sys
import os

class MaSTTest1(unittest.TestCase):
    def test_output_with_input_array(self):
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'constructNetwork.py'),
            '-i',
            '[[0,1,5,0,9],[0,0,2,7,0],[0,0,0,3,1],[0,0,0,0,14],[0,0,0,0,0]]'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        expected_output = '''
Graph:
[[ 0  1  5  0  9]
 [ 0  0  2  7  0]
 [ 0  0  0  3  1]
 [ 0  0  0  0 14]
 [ 0  0  0  0  0]]

MaST:
[[ 0.  0.  5.  0.  9.]
 [ 0.  0.  0.  7.  0.]
 [ 0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0. 14.]
 [ 0.  0.  0.  0.  0.]]
'''
        
        self.assertEqual(result.stdout.strip(), expected_output.strip())

class MaSTTest2(unittest.TestCase):
    def test_output_with_input_array(self):
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'constructNetwork.py'),
            '-i',
            '[[0,1,5,0,9],[0,0,2,7,0],[0,0,0,3,1],[0,0,0,0,0],[0,0,0,0,0]]'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        expected_output = '''
Graph:
[[0 1 5 0 9]
 [0 0 2 7 0]
 [0 0 0 3 1]
 [0 0 0 0 0]
 [0 0 0 0 0]]

MaST:
[[0. 0. 5. 0. 9.]
 [0. 0. 0. 7. 0.]
 [0. 0. 0. 3. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
'''
        self.assertEqual(result.stdout.strip(), expected_output.strip())

class MaSTTest3(unittest.TestCase):
#Case where input is already a MiST and MaST. Thus this will pass for both a MiST and MaST
    def test_output_with_input_array(self):
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'constructNetwork.py'),
            '-i',
            '[[0,0,5,0,9],[0,0,0,7,0],[0,0,0,3,0],[0,0,0,0,0],[0,0,0,0,0]]'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        expected_output = '''
Graph:
[[0 0 5 0 9]
 [0 0 0 7 0]
 [0 0 0 3 0]
 [0 0 0 0 0]
 [0 0 0 0 0]]

MaST:
[[0. 0. 5. 0. 9.]
 [0. 0. 0. 7. 0.]
 [0. 0. 0. 3. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
'''
        
        self.assertEqual(result.stdout.strip(), expected_output.strip())


class MaSTTest4(unittest.TestCase):
    def test_output_with_input_array(self):
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'constructNetwork.py'),
            '-i',
            '[[0,2,5],[0,0,1],[0,0,0]]'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        expected_output = '''
Graph:
[[0 2 5]
 [0 0 1]
 [0 0 0]]

MaST:
[[0. 2. 5.]
 [0. 0. 0.]
 [0. 0. 0.]]
'''
        
        self.assertEqual(result.stdout.strip(), expected_output.strip())

if __name__ == '__main__':
    unittest.main()