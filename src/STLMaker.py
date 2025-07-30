import numpy as np
import stl
from stl import mesh


class Mesh2STL:
    @staticmethod
    def mesh_to_stl(
        X, Y, Z, N, file_name, unit_in="m", unit_out="mm", binary=True
    ):
        """
        Inputs:
            X: 2D numpy array of values defining the x position of each pixel
            ie for a 1x1m sized mesh object with a 0.1m resolution we expect
            X = [[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
                [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
                                    ...
                [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
                [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]]

            Y: 2D numpy array of values defining the y position of each pixel
            ie for a 1x1m sized mesh object with a 0.1m resolution we expect
            Y = [[[0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
                                    ...
                [0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8]
                [0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9]]

            Z: 2D numpy array of values defining the z position (height) of
            each pixel using plt.imshow(Z) should yeild a 'surface map' type
            image

            N: The number of pixels in the array, ie X, Y, Z should all be NxN

            file_name: The path/name of the file name to save under

            unit_in: The input units of the X, Y, Z arrays. Default is meters.
            Can be 'm', 'mm', 'um', 'nm'

            unit_out: The output units of the stl file. ie for a 1x1m sized
            mesh object with units in mm we expect the X, Y output to range
            from 0 to 900. Default is milimeters. Can be 'm', 'mm', 'um', 'nm'

            binary: If True file will be output as binary (smaller), else
            ASCII (readable)

        Output:
            returns None, saves an stl file to disk

        -------------------------------------------------------------------
        Algorithm:
            Method: Decompose the traversal pattern into a series of simpler
            patterns and combine them to get the final pattern.

            For a gridsize of N, we have N-1 squares in each row/column,
            giving a total of (N-1)^2 sqaures and 2*(N-1)^2 triangles

            Each triangle is composed of three points and a so a square is
            made of 6 points

            Within the series of points there is a repeating pattern which can
            be decomposed into thee different sequences and combines

            S1: [0,0,0], [1,1,1], [1,1,1], [2,2,2], [2,2,2], [3,3,3], ...
            S2: [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], ...
            S3: [N,N,N], [0,0,0], [N,N,N], [0,0,0], [N,N,N], [0,0,0], ...
            Combining these together gives us our repeating pattern,
            S = S1 + S2 + S3

            Note: These brackes are there to help define the repeating
            sequence and do not relate to the final grouping into triangles.

            The final sequence is then [0, 1, S, N-1] and represents a full
            row of triangles. This final sequence is then repeated again
            with N added to each value, then again with 2*N added to each
            value and so on.
        """

        Z = np.array(Z)

        # Dictionary of indexes for scaling values
        dic = {
            "m": 0,
            "mm": 1,
            "um": 2,
            "nm": 3,
        }

        # Array of relevant scaling values
        # X axis is unit in, Y axis is unit out
        scale_arr = [
            [1, 1e-3, 1e-6, 1e-9],
            [1e3, 1, 1e-3, 1e-6],
            [1e6, 1e3, 1, 1e-3],
            [1e9, 1e6, 1e3, 1],
        ]

        points = (
            np.array([X.flatten(), Y.flatten(), Z.flatten()])
            * scale_arr[dic[unit_out]][dic[unit_in]]
        )

        # Other Numer of trianges and squares
        squares_per_row = N - 1
        num_squares = squares_per_row**2
        num_triangles = 2 * num_squares

        # Points per triangle and square
        points_per_triangle = 3
        points_per_square = 2 * points_per_triangle

        # Number of points per row
        points_per_row = squares_per_row * points_per_square

        # Size of the sequences
        sequence_size = points_per_row - 3
        full_sequence_size = sequence_size + 3

        # Create indexes used to create sequences
        i = np.linspace(0, sequence_size, num=sequence_size, endpoint=False)
        j = np.floor_divide(i, 3)

        # Sequence 1
        S1 = np.floor_divide(j + 1, 2)

        # Sequence 2
        S2_mask = (i + 1) % 3 == 0
        S2 = np.ones(sequence_size) * S2_mask

        # Sequence 3
        S3_mask = j % 2 == 0
        S3 = N * np.ones(sequence_size) * S3_mask

        # Add together
        S = S1 + S2 + S3

        # Non repeating start and end of array
        start = np.array([0, 1])
        end = np.array([squares_per_row])

        # Combine
        sequence = np.concatenate([start, S, end])
        base_sequence = np.tile(sequence, reps=squares_per_row)

        # Create meshgrid of integer N values for the different rows
        arr_x = np.linspace(
            0, full_sequence_size, num=full_sequence_size, endpoint=False
        )
        arr_y = np.linspace(
            0, squares_per_row, num=squares_per_row, endpoint=False
        )
        X, Y = np.meshgrid(arr_x, arr_y)
        incremental_sequence = (
            N * Y.flatten()
        )  # Flatten here in order to combine

        # Combine sequences together to create final array
        full_sequence_flat = base_sequence + incremental_sequence
        full_sequence = full_sequence_flat.reshape([num_triangles, 3])

        # Force integer values
        full_sequence = np.array(full_sequence, dtype=int)

        # Create the vertices and points arrays
        vertices = points.T
        faces = full_sequence

        # Create the mesh object
        grid_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        # input mesh vectors
        for i, f in enumerate(faces):
            for j in range(3):
                grid_mesh.vectors[i][j] = vertices[f[j], :]

        # Write the mesh to file
        if binary:
            grid_mesh.save("{}.stl".format(file_name))
        else:
            grid_mesh.save("{}.stl".format(file_name), mode=stl.Mode.ASCII)
