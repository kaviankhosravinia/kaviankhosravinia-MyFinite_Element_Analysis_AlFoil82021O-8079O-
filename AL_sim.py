from mesh_from_polygon_general import mesh_polygon, visualize_delaunay
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from scipy.spatial import Delaunay

# ================================================
E, nu, rho = 69e3, 0.33, 2700  # Young's modulus (Pa), Poisson's ratio, density (kg/m³)

mesh_params = {
"circle_radius": 0.1,
"max_Area": 0.005,
"max_angle": 10,
"grid_size_x": 150,
"grid_size_y": 8,
"grid_size_r": 32,
"grid_size_t": 20,
"segment_length": 0.002}



total_force=1657.9*9.81/1000
# Define the part
polygon_points = np.array([(0, 0), (10, 0), (10, 0.183), (0, 0.183), (0, 0)])
touch_point = np.array([5.0, 0.183])  # Contact point (x, y)
circle_center = Point(touch_point)


mesh_part, subdivided_polygon0=mesh_polygon(polygon_points,circle_center,mesh_params)
visualize_delaunay(mesh_part)


# debris polyline
theta0 = np.linspace(-np.pi/4, 5/4*np.pi, 20, endpoint=False)

theta1 = np.linspace(5/4*np.pi, 7/4*np.pi, 40, endpoint=False)


theta = np.concatenate((theta0, theta1))
print("theta",theta)

radius=0.9
x = radius * np.cos(theta)
y = radius * np.sin(theta)
polyline = np.column_stack((x+5.0, y+0.183+radius))
polylineD = np.vstack([polyline, polyline[0]])


mesh_params = {
"circle_radius": 0.05,
"grid_size_x": 4,
"grid_size_y": 4,
"grid_size_r": 1,
"grid_size_t": 0,
"segment_length": 0.01}
touch_point = np.array([5.0, 0.183+radius])  # Contact point (x, y)
circle_center = Point(touch_point)

mesh_debris, subdivided_polygon0=mesh_polygon(polylineD,circle_center,mesh_params)
visualize_delaunay(mesh_debris)
plt.show()




def update_contact_nodes_and_positions_with_polyline(points, polyline_vertices):
    """
    Identifies points inside a polyline, projects them onto the nearest edge,
    and returns updated points, contact nodes, and displacements.
    """

    def is_point_inside_polyline(point, polyline):
        """Uses winding number to check if a point is inside a polygon."""
        wn = 0  # Winding number counter
        for i in range(len(polyline) - 1):
            if polyline[i][1] <= point[1]:
                if polyline[i+1][1] > point[1]:  # Upward crossing
                    if is_left(polyline[i], polyline[i+1], point) > 0:
                        wn += 1
            else:
                if polyline[i+1][1] <= point[1]:  # Downward crossing
                    if is_left(polyline[i], polyline[i+1], point) < 0:
                        wn -= 1
        return wn != 0

    def is_left(p0, p1, p2):
        """Helper function to determine if a point is to the left of a line."""
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

    def project_point_to_edge(point, edge_start, edge_end):
        """Projects a point onto a line segment."""
        edge_vector = edge_end - edge_start
        t = np.dot(point - edge_start, edge_vector) / np.dot(edge_vector, edge_vector)
        t = np.clip(t, 0, 1)
        return edge_start + t * edge_vector

    contact_nodes = []
    displacements = []
    updated_points = points.copy()
    min_distances = []

    for i, point in enumerate(points):
        if is_point_inside_polyline(point, polyline_vertices):
            min_dist = float('inf')
            min_proj = None
            for j in range(len(polyline_vertices) - 1):
                proj = project_point_to_edge(point, polyline_vertices[j], polyline_vertices[j+1])
                dist = np.linalg.norm(point - proj)
                if dist < min_dist:
                    min_dist = dist
                    min_proj = proj
            contact_nodes.append(i)
            displacements.append(min_proj - point)
            updated_points[i] = min_proj
            min_distances.append(min_dist)
        else:
          min_dist = float('inf')
          for j in range(len(polyline_vertices) - 1):
                proj = project_point_to_edge(point, polyline_vertices[j], polyline_vertices[j+1])
                dist = np.linalg.norm(point - proj)
                if dist < min_dist:
                    min_dist = dist
          min_distances.append(min_dist)

    return updated_points, contact_nodes, displacements, min_distances

def element_matrices(coords, E, nu, rho):
    # Compute triangle area
    x1, y1, x2, y2, x3, y3 = coords.flatten()
    A = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Material matrix D for plane stress
    D = (E / (1 - nu**2)) * np.array([[1, nu, 0],
                                       [nu, 1, 0],
                                       [0,  0, (1 - nu) / 2]])
    
    # Compute inverse Jacobian matrix
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    inv_J = np.vstack((b, c)) / (2 * A)

    # Compute B matrix
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, i * 2] = inv_J[0, i]
        B[1, i * 2 + 1] = inv_J[1, i]
        B[2, i * 2:i * 2 + 2] = inv_J[:, i]
    
    # Stiffness matrix
    K_e = A * B.T @ D @ B

    # Lumped mass matrix expanded to 6x6
    M_e_full = (rho * A / 6) * np.kron(np.eye(3), np.array([[2, 0], [0, 2]]))

    return K_e, M_e_full


def assemble_global_matrix(K_global, K_e, el):

    # Map local DOFs (degrees of freedom) to global DOFs
    dof_map = []
    for node in el:
        dof_map.extend([node * 2, node * 2 + 1])  # Each node has 2 DOFs (x, y)

    # Assemble K_e into K_global
    for i in range(len(dof_map)):
        for j in range(len(dof_map)):
            K_global[dof_map[i], dof_map[j]] += K_e[i, j]
    return K_global


def plot_points_and_polyline(points, polyline_vertices, updated_points=None):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the original points
    ax.scatter(points[:, 0], points[:, 1], color='blue', label='Original Points', s=10, alpha=0.7)
    
    # Plot the polyline
    ax.plot(polyline_vertices[:, 0], polyline_vertices[:, 1], color='red', label='Polyline', linewidth=2)
    
    # If updated points are provided, plot them
    if updated_points is not None:
        ax.scatter(updated_points[:, 0], updated_points[:, 1], color='green', label='Updated Points', s=50, alpha=0.7)
        # Draw displacement vectors
        for original, updated in zip(points, updated_points):
            ax.arrow(original[0], original[1], 
                     updated[0] - original[0], updated[1] - original[1], 
                     head_width=0.2, head_length=0.2, color='gray', alpha=0.5, length_includes_head=True)
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Deformed Geometry')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')


points=np.array(mesh_part.points)
cells=np.array(mesh_part.simplices)

# Identify boundary nodes
fixed_nodes = np.where(points[:, 1] == 0)[0]  # Nodes on the bottom side (y = 0)
fixed_dofs=[]
dof = points.shape[0] * 2
u = np.zeros(dof)  # Displacement
F_f = np.zeros(dof)




dt, T = 0.01, 0.5 
time_steps = int(T / dt)
# debris movement
y_translation=-0.0003
total_internal_force=0
total_force=0
points0=points
contact_dofs=[]
u_t=u
depth=0
# Time-stepping loop
h=0
polylineD0=polylineD
for step in range(time_steps):
    t = step * dt
    polylineD=polylineD+ np.array([0,y_translation])

    updated_points, contact_nodes, displacements, min_distances = update_contact_nodes_and_positions_with_polyline(points, polylineD)
    #plot_points_and_polyline(points, polylineD)
    # Initialize global matrices
    K_global = np.zeros((dof, dof))

    displacements = np.array(displacements)
    
    contact_nodes = np.array(contact_nodes)

        
    if len(contact_nodes) >= 2 and h==0:
        print("contact_nodes , t", [contact_nodes, t])
        # Identify boundary nodes
        u=0*u
        free_nodes = []

        depth +=y_translation
        print("depth:",depth)
        polylineD0=polylineD
        # print("fixed_nodes", fixed_nodes)
        free_nodes = np.setdiff1d(np.arange(len(points)), np.union1d(fixed_nodes, contact_nodes))
        free_dofs = []
        for node in free_nodes:
            free_dofs.extend([node * 2, node * 2 + 1])  # Two DOFs per node
        

        # Apply boundary conditions (Fixed nodes)
        u[2 * fixed_nodes] = 0         # u_x = 0 for fixed nodes
        u[2 * fixed_nodes + 1] = 0     # u_y = 0 for fixed nodes

        #u[2 * moving_nodes] += v0 * dt # Update displacement for moving nodes
        u[2 * contact_nodes] = displacements[:, 0]
        u[2 * contact_nodes + 1] = displacements[:, 1]
        
        
        for i, el in enumerate(cells):
            coords = points[el]
            K_e, _ = element_matrices(coords, E, nu, rho)
            K_global=assemble_global_matrix(K_global, K_e, el)

            # Extract free DOFs for iterative update
        K_free = K_global[np.ix_(free_dofs, free_dofs)]
        F_free = F_f[free_dofs]  # Force vector for free DOFs
        #print("F_free",F_free)
        
       
        itii=1
        # Compute residual forces
        for iii in range(itii): 
            F_int = K_global.dot(u)

            residual = F_free - F_int[free_dofs]  # External - internal forces
            residual_norm = np.linalg.norm(residual)
            # print("iteration:", iii)
            print("residual_norm", residual_norm)
            # Solve for displacement increment
            delta_u = np.linalg.solve(K_free, residual)
            u[free_dofs] += delta_u
        
        
        F_f[free_dofs] =F_int[free_dofs]
        #total_force = np.sum(F_cy)
        #print("F_cy:",total_force)
        #print("F_f[free_dofs]", F_f[free_dofs])
        
        total_disps=np.column_stack([u[::2], u[1::2]])
        #u_prev =u
        points=points+total_disps
        total_disps=points-points0
        u_t[::2]=total_disps[:, 0]
        u_t[1::2]=total_disps[:, 1]

        F_c = K_global.dot(u_t)
        F_cx= F_c[2 * fixed_nodes]
        F_cy= F_c[2 * fixed_nodes + 1]

        F_f[free_dofs] =F_int[free_dofs]
        total_force = np.sum(F_cy)
        print("F_cy:",total_force)
        if total_force>16.1:
            h=1
            break
        
        #print("u", u)



print("F_cy:",total_force)

plot_points_and_polyline(points, polylineD0)
mesh_part1=mesh_part


plt.triplot(points[:, 0], points[:, 1], mesh_part1.simplices)
plt.show()
