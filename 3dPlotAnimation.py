import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from mpl_toolkits.mplot3d import axes3d, art3d
from matplotlib.animation import PillowWriter

muscle = np.genfromtxt("JointPositionExercise1Filtered.csv", delimiter = ',', skip_header = 1)

xMax, xMin = 0, 0
yMax, yMin = 0, 0
zMax, zMin = 0, 0

for k in range(4, 99, 4):
    muscle[:, k] = muscle[:, k] - np.median(muscle[:,0])
    muscle[:, k+1] = muscle[:, k+1] - np.median(muscle[:,1])
    muscle[:, k+2] = muscle[:, k+2] - np.median(muscle[:,2])

    # Getting Max and Min for x y z coordinates
    xMax = max(xMax, max(muscle[:, k]))
    xMin = min(xMin, min(muscle[:, k]))
    yMax = max(yMax, max(muscle[:, k+2]))
    yMin = min(yMin, min(muscle[:, k+2])) 
    zMax = max(zMax, max(muscle[:, k+1]))
    zMin = min(zMin, min(muscle[:, k+1])) 

# Swapping y and Z axis 
for k in range(0, 99, 4):
    muscle[:, [k+1, k+2]] = muscle[:, [k+2, k+1]]

muscle[:, 0] = muscle[:, 0] - np.median(muscle[:, 0])
muscle[:, 1] = muscle[:, 1] - np.median(muscle[:, 1])
muscle[:, 2] = muscle[:, 2] - np.median(muscle[:, 2])

fig = plt.figure()
ax = axes3d.Axes3D(fig)

# index_Spine_Base=1 # index_Spine_Mid=5 # index_Neck=9 # index_Head=13 
# index_Shoulder_Left=17 # index_Elbow_Left=21 # index_Wrist_Left=25 # index_Hand_Left=29 
# index_Shoulder_Right=33 # index_Elbow_Right=37 # index_Wrist_Right=41 # index_Hand_Right=45 
# index_Hip_Left=49 
# index_Knee_Left=53 # index_Ankle_Left=57 # index_Foot_Left=61 
# index_Hip_Right=65
# index_Knee_Right=69 # index_Ankle_Right=73 # index_Foot_Right=77 # index_Spine_Shoulder=81
# index_Tip_Left=85 # index_Thumb_Left=89 # index_Tip_Right=93 # index_Thumb_Right=97

# Middle part of body, head, spine, and hip
xHead = np.array(np.transpose(muscle[:, 12]))
yHead = np.array(np.transpose(muscle[:, 13]))
zHead = np.array(np.transpose(muscle[:, 14]))
dataHead = np.vstack((xHead, yHead, zHead))

xSpineMid = np.array(np.transpose(muscle[:, 4]))
ySpineMid = np.array(np.transpose(muscle[:, 5]))
zSpineMid = np.array(np.transpose(muscle[:, 6]))
dataSpineMid = np.vstack((xSpineMid, ySpineMid, zSpineMid))

xSpineBase = np.array(np.transpose(muscle[:, 0]))
ySpineBase = np.array(np.transpose(muscle[:, 1]))
zSpineBase = np.array(np.transpose(muscle[:, 2]))
dataSpineBase = np.vstack((xSpineBase, ySpineBase, zSpineBase))

xLeftHip = np.array(np.transpose(muscle[:, 48]))
yLeftHip = np.array(np.transpose(muscle[:, 49]))
zLeftHip = np.array(np.transpose(muscle[:, 50]))
dataLeftHip = np.vstack((xLeftHip, yLeftHip, zLeftHip))

xRightHip = np.array(np.transpose(muscle[:, 64]))
yRightHip = np.array(np.transpose(muscle[:, 65]))
zRightHip = np.array(np.transpose(muscle[:, 66]))
dataRightHip = np.vstack((xRightHip, yRightHip, zRightHip))

# Left side of body, hand, elbow, shoulder, knee, ankle
xLeftHand = np.array(np.transpose(muscle[:,28]))
yLeftHand = np.array(np.transpose(muscle[:,29]))
zLeftHand = np.array(np.transpose(muscle[:,30]))
dataLeftHand = np.vstack((xLeftHand, yLeftHand, zLeftHand))

xLeftElbow = np.array(np.transpose(muscle[:, 20]))
yLeftElbow = np.array(np.transpose(muscle[:, 21]))
zLeftElbow = np.array(np.transpose(muscle[:, 22]))
dataLeftElbow = np.vstack((xLeftElbow, yLeftElbow, zLeftElbow))

xLeftShoulder = np.array(np.transpose(muscle[:, 16]))
yLeftShoulder = np.array(np.transpose(muscle[:, 17]))
zLeftShoulder = np.array(np.transpose(muscle[:, 18]))
dataLeftShoulder = np.vstack((xLeftShoulder, yLeftShoulder, zLeftShoulder))

xLeftKnee = np.array(np.transpose(muscle[:, 52]))
yLeftKnee = np.array(np.transpose(muscle[:, 53]))
zLeftKnee = np.array(np.transpose(muscle[:, 54]))
dataLeftKnee = np.vstack((xLeftKnee, yLeftKnee, zLeftKnee))

xLeftAnkle = np.array(np.transpose(muscle[:, 56]))
yLeftAnkle = np.array(np.transpose(muscle[:, 57]))
zLeftAnkle = np.array(np.transpose(muscle[:, 58]))
dataLeftAnkle = np.vstack((xLeftAnkle, yLeftAnkle, zLeftAnkle))

# Right side of body, hand, elbow, shoulder, knee, ankle
xRightHand = np.array(np.transpose(muscle[:,44]))
yRightHand = np.array(np.transpose(muscle[:,45]))
zRightHand = np.array(np.transpose(muscle[:,46]))
dataRightHand = np.vstack((xRightHand, yRightHand, zRightHand))

xRightElbow = np.array(np.transpose(muscle[:, 36]))
yRightElbow = np.array(np.transpose(muscle[:, 37]))
zRightElbow = np.array(np.transpose(muscle[:, 38]))
dataRightElbow = np.vstack((xRightElbow, yRightElbow, zRightElbow))

xRightShoulder = np.array(np.transpose(muscle[:, 32]))
yRightShoulder = np.array(np.transpose(muscle[:, 33]))
zRightShoulder = np.array(np.transpose(muscle[:, 34]))
dataRightShoulder = np.vstack((xRightShoulder, yRightShoulder, zRightShoulder))

xRightKnee = np.array(np.transpose(muscle[:, 68]))
yRightKnee = np.array(np.transpose(muscle[:, 69]))
zRightKnee = np.array(np.transpose(muscle[:, 70]))
dataRightKnee = np.vstack((xRightKnee, yRightKnee, zRightKnee))

xRightAnkle = np.array(np.transpose(muscle[:, 72]))
yRightAnkle = np.array(np.transpose(muscle[:, 73]))
zRightAnkle = np.array(np.transpose(muscle[:, 74]))
dataRightAnkle = np.vstack((xRightAnkle, yRightAnkle, zRightAnkle))

# Setting labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Setting limits for x y z
# Subtracting and adding a small number to give it more space
xMin -= 0.1
yMin -= 0.3
# zMin -= 0.1
xMax += 0.1
yMax += 0.2
zMax += 0.1
ax.set_xlim3d([xMin, xMax])
ax.set_ylim3d([yMin, yMax])
ax.set_zlim3d([zMin, zMax])

# Initializing left side plot points with blue dots
LeftHandPoint, = ax.plot(dataLeftHand[0, 0:1], dataLeftHand[1,0:1], dataLeftHand[2, 0:1], 'bo')

LeftElbowPoint, = ax.plot(dataLeftElbow[0, 0:1], dataLeftElbow[1, 0:1], dataLeftElbow[2, 0:1], 'bo')

LeftShoulderPoint, = ax.plot(dataLeftShoulder[0, 0:1], dataLeftShoulder[1, 0:1], dataLeftShoulder[2, 0:1], 'bo')

LeftKneePoint, = ax.plot(dataLeftKnee[0, 0:1], dataLeftKnee[1, 0:1], dataLeftKnee[2, 0:1], 'bo')

LeftAnklePoint, = ax.plot(dataLeftAnkle[0, 0:1], dataLeftAnkle[1,0:1], dataLeftAnkle[2, 0:1], 'bo')

# Initializing right side plot points with red dots
RightHandPoint, = ax.plot(dataRightHand[0, 0:1], dataRightHand[1, 0:1], dataRightHand[2, 0:1], 'ro')

RightElbowPoint, = ax.plot(dataRightElbow[0, 0:1], dataRightElbow[1, 0:1], dataRightElbow[2, 0:1], 'ro')

RightShoulderPoint, = ax.plot(dataRightShoulder[0, 0:1], dataRightShoulder[1, 0:1], dataRightShoulder[2, 0:1], 'ro')

RightKneePoint, = ax.plot(dataRightKnee[0, 0:1], dataRightKnee[1, 0:1], dataRightKnee[2, 0:1], 'ro')

RightAnklePoint, = ax.plot(dataRightAnkle[0, 0:1], dataRightAnkle[1,0:1], dataRightAnkle[2, 0:1], 'ro')

# Initializing middle part plot points with yellow dots
Head, = ax.plot(dataHead[0, 0:1], dataHead[1, 0:1], dataHead[2, 0:1], 'yo', markersize = 12)
SpineMid, = ax.plot(dataSpineMid[0, 0:1], dataSpineMid[1, 0:1], dataSpineMid[2, 0:1], 'yo')
SpineBase, = ax.plot(dataSpineBase[0, 0:1], dataSpineBase[1, 0:1], dataSpineBase[2, 0:1], 'yo')
LeftHipPoint, = ax.plot(dataLeftHip[0, 0:1], dataLeftHip[1, 0:1], dataLeftHip[2, 0:1], 'yo')
RightHipPoint, = ax.plot(dataRightHip[0, 0:1], dataRightHip[1, 0:1], dataRightHip[2, 0:1], 'yo')

# Total Frame Count by total datapoints
animationFrameCount = int(xHead.shape[0])
animationFrameCount -=1

# Animation Lines
# Initialize a 3D line with art3d.Line3D, then adding the line onto the plot with ax.add_line
LeftHandToElbow = art3d.Line3D([dataLeftHand[0, 0], dataLeftElbow[0, 0]], [dataLeftHand[1,0], dataLeftElbow[1, 0]], [dataLeftHand[2, 0], dataLeftElbow[2, 0]])
ax.add_line(LeftHandToElbow)

RightHandToElbow = art3d.Line3D([dataRightHand[0, 0], dataRightElbow[0, 0]], [dataRightHand[1,0], dataRightElbow[1, 0]], [dataRightHand[2, 0], dataRightElbow[2, 0]])
ax.add_line(RightHandToElbow)

LeftElbowToShoulder = art3d.Line3D([dataLeftElbow[0, 0], dataLeftShoulder[0, 0]], [dataLeftElbow[1,0], dataLeftShoulder[1, 0]], [dataLeftElbow[2, 0], dataLeftShoulder[2, 0]])
ax.add_line(LeftElbowToShoulder)

RightElbowToShoulder = art3d.Line3D([dataRightElbow[0, 0], dataRightShoulder[0, 0]], [dataRightElbow[1,0], dataRightShoulder[1, 0]], [dataRightElbow[2, 0], dataRightShoulder[2, 0]])
ax.add_line(RightElbowToShoulder)

ShoulderToShoulder = art3d.Line3D([dataLeftShoulder[0, 0], dataRightShoulder[0, 0]], [dataLeftShoulder[1,0], dataRightShoulder[1, 0]], [dataLeftShoulder[2, 0], dataRightShoulder[2, 0]])
ax.add_line(ShoulderToShoulder)

HeadToSpineMid = art3d.Line3D([dataHead[0, 0], dataSpineMid[0, 0]], [dataHead[1,0], dataSpineMid[1, 0]], [dataHead[2, 0], dataSpineMid[2, 0]])
ax.add_line(HeadToSpineMid)

SpineMidToSpineBase = art3d.Line3D([dataSpineMid[0, 0], dataSpineBase[0, 0]], [dataSpineMid[1,0], dataSpineBase[1, 0]], [dataSpineMid[2, 0], dataSpineBase[2, 0]])
ax.add_line(SpineMidToSpineBase)

SpineBaseToLeftHip = art3d.Line3D([dataSpineBase[0, 0], dataLeftHip[0, 0]], [dataSpineBase[1,0], dataLeftHip[1, 0]], [dataSpineBase[2, 0], dataLeftHip[2, 0]])
ax.add_line(SpineBaseToLeftHip)

SpineBaseToRightHip = art3d.Line3D([dataSpineBase[0, 0], dataRightHip[0, 0]], [dataSpineBase[1,0], dataRightHip[1, 0]], [dataSpineBase[2, 0], dataRightHip[2, 0]])
ax.add_line(SpineBaseToRightHip)

LeftHipToLeftKnee = art3d.Line3D([dataLeftHip[0,0], dataLeftKnee[0,0]],[dataLeftHip[1,0], dataLeftKnee[1,0]],[dataLeftHip[2,0],dataLeftKnee[2,0]])
ax.add_line(LeftHipToLeftKnee)

RightHipToRightKnee = art3d.Line3D([dataRightHip[0,0],dataRightKnee[0,0]], [dataRightHip[1, 0], dataRightKnee[1,0]], [dataRightHip[2,0], dataRightKnee[2,0]])
ax.add_line(RightHipToRightKnee)

LeftKneeToLeftAnkle = art3d.Line3D([dataLeftKnee[0,0], dataLeftAnkle[0,0]], [dataLeftKnee[1,0], dataLeftAnkle[1,0]], [dataLeftKnee[2,0], dataLeftAnkle[2,0]])
ax.add_line(LeftKneeToLeftAnkle)

RightKneeToRightAnkle = art3d.Line3D([dataRightKnee[0,0], dataRightAnkle[0,0]], [dataRightKnee[1,0], dataRightAnkle[1,0]], [dataRightKnee[2,0],dataRightAnkle[2,0]])
ax.add_line(RightKneeToRightAnkle)

def animationFrame(i, dataLeftHand, dataRightHand, dataLeftElbow, dataRightElbow, dataLeftShoulder, dataRightShoulder, 
    dataLeftKnee, dataRightKnee, dataLeftAnkle, dataRightAnkle,
    dataHead, dataSpineMid, dataSpineBase, dataLeftHip, dataRightHip,
    LeftHandPoint, RightHandPoint, LeftElbowPoint, RightElbowPoint, LeftShoulderPoint, RightShoulderPoint, 
    LeftKneePoint, RightKneePoint, LeftAnklePoint, RightAnklePoint,
    Head, SpineMid, SpineBase, LeftHipPoint, RightHipPoint,
    LeftHandToElbow, RightHandToElbow, LeftElbowToShoulder, RightElbowToShoulder, ShoulderToShoulder,
    HeadToSpineMid, SpineMidToSpineBase, SpineBaseToLeftHip, SpineBaseToRightHip, 
    LeftHipToLeftKnee, RightHipToRightKnee, LeftKneeToLeftAnkle, RightKneeToRightAnkle
    ):
    
    # Updating process for the points
    # First set x and y to new points from data
    # Then set z property with set_3d_properties

    # Left Side Update
    LeftHandPoint.set_data(dataLeftHand[:2, i])
    LeftHandPoint.set_3d_properties(dataLeftHand[2, i])

    LeftElbowPoint.set_data(dataLeftElbow[:2, i])
    LeftElbowPoint.set_3d_properties(dataLeftElbow[2, i])
        
    LeftShoulderPoint.set_data(dataLeftShoulder[:2, i])
    LeftShoulderPoint.set_3d_properties(dataLeftShoulder[2,i])

    LeftKneePoint.set_data(dataLeftKnee[:2, i])
    LeftKneePoint.set_3d_properties(dataLeftKnee[2, i])

    LeftAnklePoint.set_data(dataLeftAnkle[:2, i])
    LeftAnklePoint.set_3d_properties(dataLeftAnkle[2, i])

    # Right Side Update
    RightHandPoint.set_data(dataRightHand[:2, i])
    RightHandPoint.set_3d_properties(dataRightHand[2, i])

    RightElbowPoint.set_data(dataRightElbow[:2, i])
    RightElbowPoint.set_3d_properties(dataRightElbow[2, i])

    RightShoulderPoint.set_data(dataRightShoulder[:2, i])
    RightShoulderPoint.set_3d_properties(dataRightShoulder[2, i])
    
    RightKneePoint.set_data(dataRightKnee[:2, i])
    RightKneePoint.set_3d_properties(dataRightKnee[2, i])

    RightAnklePoint.set_data(dataRightAnkle[:2, i])
    RightAnklePoint.set_3d_properties(dataRightAnkle[2, i])

    # Middle Part Update
    Head.set_data(dataHead[:2, i])
    Head.set_3d_properties(dataHead[2,i])
    
    SpineMid.set_data(dataSpineMid[:2, i])
    SpineMid.set_3d_properties(dataSpineMid[2, i])

    SpineBase.set_data(dataSpineBase[:2, i])
    SpineBase.set_3d_properties(dataSpineBase[2, i])

    LeftHipPoint.set_data(dataLeftHip[:2, i])
    LeftHipPoint.set_3d_properties(dataLeftHip[2, i])

    RightHipPoint.set_data(dataRightHip[:2, i])
    RightHipPoint.set_3d_properties(dataRightHip[2, i])

    # Animation Line Updates
    # set_data_3d sets the new line with the new starting and ending points
    LeftHandToElbow.set_data_3d([dataLeftHand[0, i], dataLeftElbow[0, i]], [dataLeftHand[1,i], dataLeftElbow[1, i]],[dataLeftHand[2, i], dataLeftElbow[2, i]])
    LeftElbowToShoulder.set_data_3d([dataLeftElbow[0, i], dataLeftShoulder[0, i]], [dataLeftElbow[1,i], dataLeftShoulder[1, i]], [dataLeftElbow[2, i], dataLeftShoulder[2, i]])
    RightHandToElbow.set_data_3d([dataRightHand[0, i], dataRightElbow[0, i]], [dataRightHand[1,i], dataRightElbow[1, i]],[dataRightHand[2, i], dataRightElbow[2, i]])
    RightElbowToShoulder.set_data_3d([dataRightElbow[0, i], dataRightShoulder[0, i]], [dataRightElbow[1,i], dataRightShoulder[1, i]], [dataRightElbow[2, i], dataRightShoulder[2, i]])

    ShoulderToShoulder.set_data_3d([dataLeftShoulder[0, i], dataRightShoulder[0, i]], [dataLeftShoulder[1,i], dataRightShoulder[1, i]], [dataLeftShoulder[2, i], dataRightShoulder[2, i]])
    HeadToSpineMid.set_data_3d([dataHead[0, i], dataSpineMid[0, i]], [dataHead[1,i], dataSpineMid[1, i]], [dataHead[2, i], dataSpineMid[2, i]])
    SpineMidToSpineBase.set_data_3d([dataSpineMid[0, i], dataSpineBase[0, i]], [dataSpineMid[1, i], dataSpineBase[1, i]], [dataSpineMid[2, i], dataSpineBase[2, i]])
    SpineBaseToLeftHip.set_data_3d([dataSpineBase[0, i], dataLeftHip[0, i]], [dataSpineBase[1,i], dataLeftHip[1, i]], [dataSpineBase[2, i], dataLeftHip[2, i]])
    SpineBaseToRightHip.set_data_3d([dataSpineBase[0, i], dataRightHip[0, i]], [dataSpineBase[1,i], dataRightHip[1, i]], [dataSpineBase[2, i], dataRightHip[2, i]])
    LeftHipToLeftKnee.set_data_3d([dataLeftHip[0,i], dataLeftKnee[0,i]],[dataLeftHip[1,i], dataLeftKnee[1,i]],[dataLeftHip[2,i],dataLeftKnee[2,i]])
    RightHipToRightKnee.set_data_3d([dataRightHip[0,i],dataRightKnee[0,i]], [dataRightHip[1, i], dataRightKnee[1,i]], [dataRightHip[2,i], dataRightKnee[2,i]])
    LeftKneeToLeftAnkle.set_data_3d([dataLeftKnee[0,i], dataLeftAnkle[0,i]], [dataLeftKnee[1,i], dataLeftAnkle[1,i]], [dataLeftKnee[2,i], dataLeftAnkle[2,i]])
    RightKneeToRightAnkle.set_data_3d([dataRightKnee[0,i], dataRightAnkle[0,i]], [dataRightKnee[1,i], dataRightAnkle[1,i]], [dataRightKnee[2,i],dataRightAnkle[2,i]])

# fargs are arguments that will be passed into the animationFrame function
# frames is being set to the animationFrameCount that we calculated previously
# interval is the number of millisecond in delay before the next frame, the higher the number, the slower the animation

animation = animation.FuncAnimation(fig, animationFrame, frames = animationFrameCount,
    fargs=(dataLeftHand, dataRightHand, dataLeftElbow, dataRightElbow ,dataLeftShoulder, dataRightShoulder, 
    dataLeftKnee, dataRightKnee, dataLeftAnkle, dataRightAnkle,
    dataHead, dataSpineMid, dataSpineBase, dataLeftHip, dataRightHip,
    LeftHandPoint, RightHandPoint, LeftElbowPoint, RightElbowPoint, LeftShoulderPoint, RightShoulderPoint,
    LeftKneePoint, RightKneePoint, LeftAnklePoint, RightAnklePoint,
    Head, SpineMid, SpineBase, LeftHipPoint, RightHipPoint,
    LeftHandToElbow, RightHandToElbow, LeftElbowToShoulder, RightElbowToShoulder, ShoulderToShoulder,
    HeadToSpineMid, SpineMidToSpineBase, SpineBaseToLeftHip, SpineBaseToRightHip,
    LeftHipToLeftKnee, RightHipToRightKnee, LeftKneeToLeftAnkle, RightKneeToRightAnkle
    ),
    interval=1 )

plt.show()

# Save animation as gif
# animation.save('patient1ex1.gif', writer = 'Pillow', fps = 60)