def distance_points_from_lines(point, origins, directions):
    '''
    Calculate the squared distances from the point to each line defined by origins + t*directions in a vectorized manner.
    '''
    # Ensure point is in the correct shape for broadcasting
    point = np.asarray(point).reshape(1, -1)
    # Calculate projections
    t_values = np.sum((point - origins) * directions, axis=1) / np.sum(directions * directions, axis=1)
    projections = origins + np.outer(t_values, directions)
    # Calculate squared distances
    squared_distances = np.sum((projections - point) ** 2, axis=1)
    return squared_distances

def find_mean_focus_point(known_camera_locations, optical_axes, initial_guess):
    '''
    Find the mean focus point by minimizing the sum of squared distances to all rays (lines).
    Vectorized version.
    '''
    # Ensure inputs are numpy arrays
    known_camera_locations = np.asarray(known_camera_locations)
    optical_axes = np.asarray(optical_axes)
    
    def objective_function(point):
        return np.sum(distance_points_from_lines(point, known_camera_locations, optical_axes))
    
    result = minimize(objective_function, initial_guess)
    return result.x