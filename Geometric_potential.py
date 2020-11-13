#!/usr/bin/env python

'''
Created on Jun 14, 2017

@author: emarch
'''

import numpy as np
import itertools as iter

from math import sqrt,cos,acos
from scipy.spatial import Delaunay, distance
from Bio.PDB import PDBParser
from Surface import Surface

class GeometricPotential(object):
    '''
    Here is reproduced the geometric potential algorithm developed by Xie and Bourne, 2007.
    DOI: doi.org/10.1186/1471-2105-8-S4-S9
    '''

    def __init__(self):
        self.protein=None
        self.calpha=None
        self.del_triangle_points=None
        self.peeled_list_triangle_points=None
        self.environmental_boundary=None
        self.planes_from_tetrahedra=[]
        self.carbons_distances_directions={}
        self.gp=[]
        
    def set_protein(self,protein):
        self.protein=protein
    
    def get_protein(self):
        return self.protein
    
    def get_alpha_carbons(self):
        return self.calpha
    
    def get_delaunay_triangles(self):
        return self.del_triangle_points
    
    def get_peeled_triangle_points(self):
        return self.peeled_list_triangle_points
    
    def get_environmental_boundary(self):
        return self.environmental_boundary
    
    def get_planes_from_tetrahedra(self):
        return self.planes_from_tetrahedra
    
    def get_carbons_distances_directions(self):
        return self.carbons_distances_directions
    
    def get_gp(self):
        return self.gp
    
    def get_geometric_potential(self):
        self.select_alpha_carbons()
        self.get_triangulation_delaunay()
        self.peel_off_tetrahedra()
        self.remove_tetrahedra_with_radius_sphere()
        self.compute_geometric_measurements()
        
    def select_alpha_carbons(self):
        parser=PDBParser(PERMISSIVE=1, QUIET=True)
        structure=parser.get_structure('protein', self.protein)
        calphas=[]
        for model in structure:
            for chain in model.get_chains():
                for res in chain.get_residues():
                    for atom in res.get_unpacked_list():
                        if 'CA' in atom.get_id():
                            calphas.append(atom.get_coord())  
        self.calpha=np.array(calphas)
    
    def get_triangulation_delaunay(self):
        self.del_triangle_points = Delaunay(self.calpha)
        
    def peel_off_tetrahedra(self):
        list_of_index=set()
        for element in self.calculate_distance(self.calpha[self.del_triangle_points.simplices]):
            if element[0] > 30.0: list_of_index.add(element[1])
        self.peeled_list_triangle_points=np.delete(self.del_triangle_points.simplices,list(list_of_index),0)
    
    def remove_tetrahedra_with_radius_sphere(self):
        list_of_index=set()
        for i,tetrahedra in enumerate(self.calpha[self.peeled_list_triangle_points]):
            radius=self.get_radius_circumscribed_sphere(distance.pdist(tetrahedra))
            if radius > 7.5: list_of_index.add(i)
        self.environmental_boundary=np.delete(self.peeled_list_triangle_points,list(list_of_index),0)
    
    def compute_geometric_measurements(self):
        for tetrahedra in self.calpha[self.environmental_boundary]:
            triangles=self.get_triangles_from_tetrahedron(tetrahedra)
            for triangle in triangles:
                self.planes_from_tetrahedra.append(self.get_plane_from_triangle_vertices(triangle[0],triangle[1],triangle[2]))
        
        for pos,carbon in enumerate(self.calpha):
            dist,vector = min(self.get_distance_from_point_to_plane(carbon,self.planes_from_tetrahedra),key=lambda x:x[0])
            self.carbons_distances_directions.update({pos:[dist,vector]})
        
        gp_raw=[]
        
        for i in range(len(self.calpha)):
            result=float()
            for j in range(i+1,len(self.calpha)):
                distance_atoms=distance.euclidean(self.calpha[i],self.calpha[j])
                if distance_atoms < 10.0:
                    direction_angle=self.get_angle_between_alpha_carbons(self.calpha[i],self.calpha[j],distance_atoms)
                    result+=self.get_neighbourhood_results_for_GP(self.carbons_distances_directions[j][0], distance_atoms, direction_angle)
            
            GP=self.carbons_distances_directions[i][0]+result
            gp_raw.append(GP)
        
        self.gp=self.normalize_geometric_potential(gp_raw)
        
    def calculate_distance(self,stuff_to_calculate):
        for i,tetrahedra in enumerate(stuff_to_calculate):
            for a in range(len(tetrahedra)):
                for b in range(a+1,len(tetrahedra)):
                    yield distance.euclidean(tetrahedra[a],tetrahedra[b]),i
    
    def get_radius_circumscribed_sphere(self, edges):
        u,v,w,W,V,U=edges
        u1 = v*v + w*w - U*U
        v1 = w*w + u*u - V*V
        w1 = u*u + v*v - W*W
        vol = sqrt(4*u*u*v*v*w*w - u*u*u1*u1 - v*v*v1*v1 - w*w*w1*w1 + u1*v1*w1) / 12.0
        s = self.triangle_area(U,V,W) + self.triangle_area(u, v, W) + self.triangle_area(U, v, w) + self.triangle_area(u, V, w)
        radius= s/(6*vol)
        return radius
        
    def triangle_area(self,a,b,c):
        '''
        Heron's formula to calculate the area by knowing the longitude of the edges of the triangle
        '''
        s = (a + b + c) / 2.0
        return sqrt(s*(s-a)*(s-b)*(s-c))
    
    def get_triangles_from_tetrahedron(self,vertices):
        iterable_list=iter.combinations(vertices,3)
        triangles=[e for e in iterable_list]
        return triangles
    
    def get_plane_from_triangle_vertices(self,p1,p2,p3):
        # These two vectors are in the plane
        v1=p3-p1
        v2=p2-p1
        
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        
        return a,b,c,d
    
    def get_distance_from_point_to_plane(self,point,planes):
        x,y,z = point
        for plane in planes:
            a,b,c,d = plane
            distance = abs(a*x + b*y + c*z + d)/sqrt(a**2 + b**2 + c**2)
            normal_vector = (a,b,c)
            yield distance,normal_vector
    
    def get_angle_between_alpha_carbons(self,calpha1,calpha2,distance_calphas):
        a=distance_calphas
        b=distance.euclidean(calpha1, np.array([0.0,0.0,0.0]))
        c=distance.euclidean(calpha2, np.array([0.0,0.0,0.0]))
        angle=acos((b**2 + c**2 - a**2)/(2*b*c))
        return angle
        
    def get_neighbourhood_results_for_GP(self,dist_env,dist_calpha,direction):
        result=(dist_env/(dist_calpha + 1.0))*(((cos(direction))+1.0)/2.0)
        return result
    
    def normalize_geometric_potential(self,geom_list):
        mindist=min(geom_list)
        maxdist=max(geom_list)
        normalizer=lambda x: ((x -mindist)/(maxdist-mindist))*100
        vfunc=np.vectorize(normalizer)
        return vfunc(geom_list)
    
    def write_triangulation_files(self):
        surf_func=Surface()
        surf_func.write_surface_mol2(self.calpha, self.del_triangle_points.simplices, '%s_non_treatet_delaunay.mol2'%(self.protein.replace(".pdb","")))
        surf_func.write_surface_mol2(self.calpha,self.peeled_list_triangle_points, '%s_peeled_triangle_points.mol2'%(self.protein.replace(".pdb","")))
        surf_func.write_surface_mol2(self.calpha, self.environmental_boundary, '%s_environmental_boundary.mol2'%(self.protein.replace(".pdb","")))
        