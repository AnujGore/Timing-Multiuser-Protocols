from objects.environment import GridTopologyEnv

thisGrid = GridTopologyEnv(3, 0.9, 0.6, True, 10)
print(thisGrid.reset()[0])