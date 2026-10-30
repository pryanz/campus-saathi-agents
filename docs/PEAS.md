# PEAS Description: Campus Saathi

## Performance Measure ($P$)

- **Primary:** Minimize total completion time for a first-time student (Total Time = Travel Time + Service Time + Queue Waiting Time).
- **Secondary:** Minimize cluster crossings between academic and hostel zones and reduce peak waiting times.

## Environment ($E$)

- **Campus Map:** Graph consisting of 11 nodes (1 entry gate, 6 academic rooms, 4 hostel rooms) connected by weighted walkways (travel times in minutes). Two distinct clusters with high-cost crossing edges.
- **Dynamic Elements:** Scripted background crowd of $N$ students following random cluster orders, creating stochastic FIFO queues at service counters.
- **Observability:** Partially observable, governed by observation settings (0: no queue info, 1: queue length in current room, 2: global queue lengths).

## Actuators ($A$)

- Traverse an edge to an adjacent room.
- Queue at a room's counter and execute the reporting step.

## Sensors ($S$)

- Current location (room/gate identifier).
- Set of completed reporting steps.
- Current simulation time (minute ticks).
- Room queue length(s) depending on the active observation setting.
