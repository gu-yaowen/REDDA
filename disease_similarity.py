'''
Title:Mesh Similarity
Paper:Predicting Drug-Disease Associations by using Similarity Constrained Matrix Factorization,2018
Type:Disease
Descripition:
For each disease, a directed acyclic graph (DAG) is constructed based on hierarchical descriptors,
in which nodes represent disease MeSH descriptors (or disease terms) and edges
represent the relationship between the current node and its ancestors.
Then compare the DAG Similarity.
Input Args:List of diseases mesh ids.
Output Shape:(len(diseases),len(diseases))
Version:2.0 developed by Yuzhouxin
What's new:Using webapi to keep data up to date.
'''

import numpy as np
import requests
import time
import csv
import pandas as pd


def get_data(disease_id):
    '''
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
    SELECT  DISTINCT ?treeid ?aid
    FROM <http://id.nlm.nih.gov/mesh>
    WHERE{mesh:D001943 meshv:treeNumber ?treeNum.
    {{?treeNum meshv:parentTreeNumber+ ?ancestorTreeNum.?ancestor meshv:treeNumber ?ancestorTreeNum.?ancestor meshv:identifier ?aid.?ancestorTreeNum rdfs:label ?treeid.
    }UNION{?treeNum rdfs:label ?treeid.mesh:D001943 meshv:identifier ?aid.}}}
    '''
    api = 'https://id.nlm.nih.gov/mesh/sparql?query=PREFIX%20rdfs%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%20PREFIX%20meshv%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2Fvocab%23%3E%20PREFIX%20mesh%3A%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%2F%3E%20SELECT%20%20DISTINCT%20%3Ftreeid%20%3Faid%20FROM%20%3Chttp%3A%2F%2Fid.nlm.nih.gov%2Fmesh%3E%20WHERE%7Bmesh%3A' + disease_id + '%20meshv%3AtreeNumber%20%3FtreeNum.%20%7B%7B%3FtreeNum%20meshv%3AparentTreeNumber%2B%20%3FancestorTreeNum.%3Fancestor%20meshv%3AtreeNumber%20%3FancestorTreeNum.%3Fancestor%20meshv%3Aidentifier%20%3Faid.%3FancestorTreeNum%20rdfs%3Alabel%20%3Ftreeid.%20%7DUNION%7B%3FtreeNum%20rdfs%3Alabel%20%3Ftreeid.mesh%3A' + disease_id + '%20meshv%3Aidentifier%20%3Faid.%7D%7D%7D&format=CSV&inference=false&offset=0&limit=1000'
    req = requests.get(api).text.split('\r\n')[1:-1]
    print(req)
    treeDic = {}
    treenumberlist = []
    for each in req:
        each = each.split(',')
        treeDic[each[0]] = each[1]
        if each[1] == disease_id:
            treenumberlist.append(each[0])
    return treeDic, treenumberlist


def construct_DAG(disease_id):
    treedic, treenumberlist = get_data(disease_id)
    # if len(treenumberlist)==0:
    #   print(disease_id)
    disease_dict = {}
    for treenum in range(len(treenumberlist)):
        nodetree = treenumberlist[treenum].split('.')
        new_nodetree = []
        for nodenum in range(len(nodetree)):
            new_node = ""
            for i in range(nodenum + 1):
                new_node = new_node + nodetree[i] + '.'
            new_node = new_node[:-1]
            new_nodetree.append(treedic[new_node])
        for nodenum in range(len(new_nodetree)):
            if new_nodetree[nodenum] in disease_dict:
                disease_dict[new_nodetree[nodenum]] = min(len(new_nodetree) - nodenum - 1,
                                                          disease_dict[new_nodetree[nodenum]])
            else:
                disease_dict[new_nodetree[nodenum]] = len(new_nodetree) - nodenum - 1
    return disease_dict


def get_DV(disease_dict, delta=0.5):
    DV = 0
    for layer in disease_dict.values():
        DV = DV + pow(delta, layer)
    return DV


def get_intersection(disease_dict1, disease_dict2, delta=0.5):
    intersection_value = 0
    for key in disease_dict1.keys():
        if key in disease_dict2:
            intersection_value = intersection_value + pow(delta, disease_dict1[key]) + pow(delta, disease_dict2[key])
    return intersection_value


def get_MeSH_Similarity(disease_dict1, disease_dict2):
    DV1 = get_DV(disease_dict1)
    DV2 = get_DV(disease_dict2)

    intersection_value = get_intersection(disease_dict1, disease_dict2)

    return intersection_value / (DV1 + DV2)


def get_Mesh_sim_mat(Disease_list):
    Mesh_sim_mat = np.zeros([len(Disease_list), len(Disease_list)], dtype=float, order='C')
    for i in range(len(Disease_list)):
        print(Disease_list[i])
        Disease_list[i] = construct_DAG(Disease_list[i])
        print(Disease_list[i])
    for i in range(len(Disease_list)):
        Dis_i = Disease_list[i]
        for j in range(i):
            Dis_j = Disease_list[j]
            Mesh_sim_mat[i][j] = get_MeSH_Similarity(Dis_i, Dis_j)
    Mesh_sim_mat = Mesh_sim_mat + Mesh_sim_mat.T + np.eye(len(Disease_list))
    return Mesh_sim_mat


def get_Disease_list():
    with open('disease.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        Disease_list = ['mesh:' + row[0] for row in reader]
        return Disease_list


# Disease_list =get_Disease_list()
# input a list contains string-type disease MeSH id 
df = pd.read_csv('.\\associations\\pathway_disease.csv')
Disease_list = list(df['Disease'].value_counts().index)
# calculate the Mesh similarity based on web search
dis_sim = get_Mesh_sim_mat(Disease_list)
np.savetxt('DisSim.txt', dis_sim)
