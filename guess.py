import random
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import datetime

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_random_points(n, x_limit=100, y_limit=100):
    return [(random.uniform(0, x_limit), random.uniform(0, y_limit)) for _ in range(n)]

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

def kruskal_mst(points, active_nodes):
    edges = [(distance(points[i], points[j]), i, j) 
             for i in range(len(points)) 
             for j in range(i + 1, len(points))]
    edges.sort()

    parent = list(range(len(points)))
    rank = [0] * len(points)
    mst = []

    for d, u, v in edges:
        if active_nodes[u] and active_nodes[v] and find(parent, u) != find(parent, v):
            mst.append((u, v, d))
            union(parent, rank, u, v)
            if len(mst) == sum(active_nodes) - 1:
                break

    return mst

def visualize_truck_route(points, demands, truck_capacity=8, origin_node=0):
    fig, (ax, ax_slider) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[9, 1])
    plt.subplots_adjust(bottom=0.2)
    
    scatter = ax.scatter(*zip(*points), color='red', zorder=5)
    texts = [ax.text(p[0], p[1], f'{i + 1} (D: {demands[i]})', fontsize=8, color='black') 
             for i, p in enumerate(points)]
    
    origin_point = points[origin_node]
    ax.scatter(*origin_point, color='blue', s=100, zorder=6)
    ax.text(origin_point[0], origin_point[1], f'Origin\n{origin_node + 1}', fontsize=10, color='blue', 
            verticalalignment='bottom', horizontalalignment='center')
    
    capacity_text = ax.text(0.02, 0.98, f'Truck Capacity: {truck_capacity}/{truck_capacity}', 
                            transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                            bbox=dict(facecolor='white', alpha=0.7))
    
    demands_sum = sum(demands)
    path_length_sum = 0
    elapsed_time = 0

    demands_text = ax.text(0.02, 0.90, f'Total Demands: {demands_sum}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    path_length_text = ax.text(0.02, 0.85, f'Total Path Length: {path_length_sum:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    time_text = ax.text(0.02, 0.80, f'Elapsed Time: {elapsed_time:.2f}s', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    current_location = origin_node
    current_capacity = truck_capacity
    pending_demands = demands[:]
    active_nodes = [True] * len(points)
    trip_count = 1
    mst_lines = []
    route = []
    truck_line = None
    start_time = datetime.datetime.now()

    def update_mst():
        nonlocal mst_lines
        for line in mst_lines:
            line.remove()
        mst_lines.clear()
        new_mst = kruskal_mst(points, active_nodes)
        for u, v, _ in new_mst:
            p1, p2 = points[u], points[v]
            line = Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='blue', alpha=0.5)
            ax.add_line(line)
            mst_lines.append(line)
        fig.canvas.draw()
        return new_mst

    def deliver_packages(vertex):
        nonlocal current_capacity, pending_demands
        if pending_demands[vertex] > 0:
            delivered = min(pending_demands[vertex], current_capacity)
            pending_demands[vertex] -= delivered
            current_capacity -= delivered
            return delivered
        return 0

    def generate_route(mst):
        adj_list = [[] for _ in range(len(points))]
        for u, v, _ in mst:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        visited = [False] * len(points)
        route = []

        def dfs(v):
            visited[v] = True
            neighbors = sorted(adj_list[v], key=lambda x: (-pending_demands[x], x))
            for neighbor in neighbors:
                if not visited[neighbor]:
                    route.append(neighbor)
                    dfs(neighbor)
                    route.append(v)

        dfs(origin_node)
        return route

    def update_affordable_nodes():
        colors = ['red' if active else 'none' for active in active_nodes]
        for i, demand in enumerate(pending_demands):
            if active_nodes[i] and demand <= current_capacity:
                colors[i] = 'green'
        scatter.set_facecolor(colors)
        scatter.set_edgecolor(colors)

    def update_delivery(frame):
        nonlocal current_location, current_capacity, trip_count, route, truck_line, points, pending_demands, active_nodes, path_length_sum, elapsed_time

        if sum(pending_demands) == 0 and not any(active_nodes):
            ax.set_title("All deliveries completed!")
            ani.event_source.stop()
            return

        if frame == 0 or len(route) == 0:
            route = generate_route(update_mst())
            current_location = origin_node
            current_capacity = truck_capacity
            trip_count += 1
            ax.set_title(f"Starting new trip (Trip {trip_count})")
            capacity_text.set_text(f'Truck Capacity: {current_capacity}/{truck_capacity}')
            update_affordable_nodes()
            return

        next_location = route.pop(0)

        while not active_nodes[next_location] and len(route) > 0:
            next_location = route.pop(0)

        if not active_nodes[next_location]:
            ani.event_source.stop()
            ax.set_title("All deliveries completed!")
            return

        if next_location >= len(points) or next_location < 0:
            return

        if truck_line:
            truck_line.remove()
        truck_line = ax.plot([points[current_location][0], points[next_location][0]], 
                             [points[current_location][1], points[next_location][1]], 'r-', linewidth=2)[0]
        
        delivered = deliver_packages(next_location)
        path_length_sum += distance(points[current_location], points[next_location])
        
        if delivered > 0:
            ax.set_title(f"Delivered {delivered} packages to vertex {next_location + 1} (Trip {trip_count})")
        else:
            ax.set_title(f"Visited vertex {next_location + 1} (Trip {trip_count})")
        
        texts[next_location].set_text(f'{next_location + 1} (D: {pending_demands[next_location]})')
        capacity_text.set_text(f'Truck Capacity: {current_capacity}/{truck_capacity}')
        demands_text.set_text(f'Total Demands: {sum(pending_demands)}')
        path_length_text.set_text(f'Total Path Length: {path_length_sum:.2f}')
        
        current_location = next_location

        if pending_demands[next_location] == 0 and next_location != origin_node:
            active_nodes[next_location] = False
            texts[next_location].set_alpha(0.3)
            if len([a for a in active_nodes if a]) == 1:
                ani.event_source.stop()
                ax.set_title("All deliveries completed!")
                return
            update_mst()

        if current_capacity == 0:
            if truck_line:
                truck_line.remove()
            truck_line = ax.plot([points[current_location][0], points[origin_node][0]], 
                                 [points[current_location][1], points[origin_node][1]], 'r-', linewidth=2)[0]
            current_location = origin_node
            current_capacity = truck_capacity
            trip_count += 1
            ax.set_title(f"Returning to origin to reload (Trip {trip_count})")
            capacity_text.set_text(f'Truck Capacity: {current_capacity}/{truck_capacity}')
            route = generate_route(update_mst())

        update_affordable_nodes()

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        time_text.set_text(f'Elapsed Time: {elapsed_time:.2f}s')

    class DraggablePoints:
        def __init__(self):
            self.selected = None
            self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
            self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        def on_press(self, event):
            if event.inaxes != ax:
                return
            for i, point in enumerate(points):
                if math.hypot(point[0] - event.xdata, point[1] - event.ydata) < 5:
                    self.selected = i
                    break

        def on_motion(self, event):
            nonlocal mst
            if self.selected is not None and event.inaxes == ax:
                points[self.selected] = (event.xdata, event.ydata)
                scatter.set_offsets(points)
                texts[self.selected].set_position((event.xdata, event.ydata))
                mst = update_mst()

        def on_release(self, event):
            self.selected = None

    mst = update_mst()

    speed_slider = Slider(ax_slider, 'Speed', 1, 100, valinit=50, valstep=1)

    ani = FuncAnimation(fig, update_delivery, frames=1000, interval=20, repeat=True, blit=False)

    def update_speed(val):
        ani.event_source.interval = 1000 / val
    speed_slider.on_changed(update_speed)

    DraggablePoints()

    plt.show()

if __name__ == "__main__":
    n = 10
    points = generate_random_points(n)
    demands = [random.randint(1, 10) for _ in range(n)]
    demands[0] = 0

    visualize_truck_route(points, demands, truck_capacity=5, origin_node=0)