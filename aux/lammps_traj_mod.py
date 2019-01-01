#!/home/mwebb/anaconda/bin/python
# This is a module that contains functions for reading information from files commonly used in LAMMPS
import imp
from numpy import *

#==================================================================
#  AUX: append_to_dict
#==================================================================
def append_to_dict(dic,key,val):
  if key in dic:
    dic[key].append(val)
  else:
    dic[key] = [val]

#==================================================================
#  AUX: init_atoms
#==================================================================
def init_atoms(N):
  atoms = {}
  atoms['coords'] = zeros([N,3])
  atoms['forces'] = zeros([N,3])
  atoms['type']   = [0]*N
  atoms['id']     = {}
  atoms['pos']    = zeros([N,3])
  atoms['charge'] = [0.0]*N
  return atoms

#==================================================================
#  AUX: get_mass_map
#==================================================================
def get_mass_map(files,atoms):

  if files['data'] != "none":
    print "# Extracting masses from ", files['data'], " ..."
    fid = open(files['data'])
    line = fid.readline().strip().split()
    while True:
      if (len(line) == 3 and line[1] == "atom" and line[2] == "types"):
        ntype = int(line[0])
        print "# A total of ", ntype, " atom types reported!!!"
      if (len(line) == 1 and line[0] == "Masses"):
        fid.readline()
        mass_map = {}
        for i in range(ntype):
          line = fid.readline().strip().split()
          mass_map[int(line[0])] = float(line[1])
        print "# Masses field found and recorded! Breaking from file..."
        fid.close()
        break
      line = fid.readline().strip().split()

  return mass_map

#==================================================================
#  AUX: get_charge_map
#==================================================================
def get_charge_map(files,atoms):
    if files['data'] != "none":
        print "# Extracting charges from ", files['data'], " ..."
        fid = open(files['data'])
        line = fid.readline().strip().split()
        qtot = 0.0
        while True:
            if (len(line) == 2 and line[1] == "atoms"):
                natm = int(line[0])
                print "# A total of ", natm, " atoms reported!!!"
            if (len(line) == 3 and line[1] == "atom" and line[2] == "types"):
                ntype = int(line[0])
                q4type= [0.0]*ntype
                n4type= [0.0]*ntype
            if (len(line) >= 1 and line[0] == "Atoms"):
                fid.readline()
                for j in range(natm):
                    line = fid.readline().strip().split()
                    ind  = int(line[0])
                    typ  = int(line[2])
                    q    = float(line[3])
                    if ind in atoms['id']:
                        ptr = atoms['id'][ind]
                        atoms['charge'][ptr] = q
                        qtot += q
                    q4type[typ-1] += q
                    n4type[typ-1] += 1.0
                fid.close()
                break
            line = fid.readline().strip().split()
    qavg = [qi/ni if ni > 0 else 0 for qi,ni in zip(q4type,n4type)]

    # create a type dictionary
    qmap = {}
    for i in range(ntype):
        qmap[i+1] = qavg[i]
    return qmap



#==================================================================
#  AUX: get_adjacency_list
#==================================================================
def get_adj_list(files,atoms):

  # EXAMINE TOPOLOGY FROM DATA FILE
  adjlist = {};
  if files['data'] != "none":
    print "# Extracting topology from ", files['data'], " ..."
    fid = open(files['data'])
    line = fid.readline().strip().split()
    while True:
      if (len(line) == 2 and line[1] == "bonds"):
        nbond=int(line[0])
        print "# A total of ", nbond, " bonds reported!!!"
      if (len(line) == 1 and line[0] == "Bonds"):
        fid.readline()
        for j in range(nbond):
          line = fid.readline().strip().split()
          bond = [int(el) for el in line]
          if bond[2] in atoms['id'].keys():
            append_to_dict(adjlist,bond[2],bond[3])
            append_to_dict(adjlist,bond[3],bond[2])
        print "# Bonds field found and recorded! Breaking from file..."
        fid.close()
        break
      line = fid.readline().strip().split()
  return adjlist

#==================================================================
#  AUX: process_frame(fid)
#==================================================================
def process_frame(fid_list,inc_list,atoms):
  #EXTRACT HEADER INFORMATION
  natm = []
  box = zeros([3,2])
  for fid in fid_list:
    for i in range(2):
      fid.readline()
    line = fid.readline().strip().split()
    natm += [int(line[0])]
    fid.readline()

    # GET BOX INFORMATION
    box[0][:] = [v for v in fid.readline().strip().split()]
    box[1][:] = [v for v in fid.readline().strip().split()]
    box[2][:] = [v for v in fid.readline().strip().split()]
    line = fid.readline().strip().split()
    line = line[2:]
    ind_id = line.index('id')
    ind_typ= line.index('type')
    ind_x  = line.index('x')
    ind_y  = line.index('y')
    ind_z  = line.index('z')
    forces_present = False
    if ('fx' in line):
        forces_present = True
        ind_fx  = line.index('fx')
        ind_fy  = line.index('fy')
        ind_fz  = line.index('fz')

  # GET ATOM INFORMATION
  L         = box[:,1] - box[:,0]
  for i,fid in enumerate(fid_list):
    for j in range(natm[i]):
      line      = fid.readline().strip().split()
      ind_j     = int(line[ind_id])
      type_j    = int(line[ind_typ])
      if (inc_list == "all" or type_j in inc_list):
        id_j      = atoms['id'][ind_j]
        atoms['coords'][id_j] = array([float(i) for i in [line[ind_x],line[ind_y],line[ind_z]]])
        if forces_present:
           atoms['forces'][id_j]  = array([float(i) for i in [line[ind_fx],line[ind_fy],line[ind_fz]]])
        else:
           atoms['forces'][id_j]  = zeros([3])

  return (atoms,L,0.5*L,box)


#==================================================================
#  AUX: skip_frame(ftraj)
#==================================================================
def skip_frame(ftraj):
  # SKIP HEADER INFO
  for i in range(2):
    ftraj.readline()
  line = ftraj.readline().strip().split()
  natm = int(line[0])
  for i in range(5+natm):
    ftraj.readline()

#==================================================================
#  AUX: screen_frame(fid)
#==================================================================
def screen_frame(traj_list,inc_list):

  # OPEN FILES
  fid_list      = [0]*len(traj_list)
  for i,f in enumerate(traj_list):
    fid_list[i]      = open(f,"r")
    fid_list[i].readline()

  #EXTRACT HEADER INFORMATION
  natm = []
  box = zeros([3,2])
  for fid in fid_list:
    for i in range(2):
      fid.readline()
    line = fid.readline().strip().split()
    natm += [int(line[0])]
    fid.readline()

    # GET BOX INFORMATION
    box[0][:] = [v for v in fid.readline().strip().split()]
    box[1][:] = [v for v in fid.readline().strip().split()]
    box[2][:] = [v for v in fid.readline().strip().split()]
    line = fid.readline().strip().split()
    line = line[2:]
    ind_id = line.index('id')
    ind_typ= line.index('type')


  # PARTIALLY INITIALIZE 'atoms' STRUCTURE
  atoms = {}
  atoms['id']     = {}
  atoms['type']   = []
  atoms['charge'] = []

  # GET ATOM INFORMATION
  L         = box[:,1] - box[:,0]
  count = 0
  for i,fid in enumerate(fid_list):
    for j in range(natm[i]):
      line      = fid.readline().strip().split()
      ind_j     = int(line[ind_id])
      type_j    = int(line[ind_typ])
      if (inc_list == "all" or type_j in inc_list):
        atoms['id'][ind_j] = count
        atoms['type']     += [type_j]
        atoms['charge']   += [0.0]
        count += 1

  # FINISH INITIALIZATION
  atoms['coords'] = zeros([count,3])
  atoms['forces'] = zeros([count,3])
  atoms['count']  = count

  # CLOSE FILES
  for i,f in enumerate(traj_list):
    fid_list[i].close()

  return atoms


#==================================================================
#  AUX: skip_frame(ftraj)
#==================================================================
def skip_frame(ftraj):
  # SKIP HEADER INFO
  for i in range(2):
    ftraj.readline()
  line = ftraj.readline().strip().split()
  natm = int(line[0])
  for i in range(5+natm):
    ftraj.readline()
