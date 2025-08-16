import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import networkx as nx

class Huckel:
    """
    Classe para implementar o método de Hückel para cálculos de estrutura eletrônica
    de sistemas π conjugados.
    """
    
    def __init__(self, n_atoms):
        """
        Inicializa o sistema com n átomos.
        
        Args:
            n_atoms (int): Número de átomos no sistema
        """
        self.n_atoms = n_atoms
        self.hamiltonian = np.zeros((n_atoms, n_atoms))
        self.atom_types = ['C'] * n_atoms  # Por padrão, todos são carbonos
        self.connectivity = []
        self.eigenvalues = None
        self.eigenvectors = None
        
        # Parâmetros padrão do método de Hückel
        self.alpha_C = 0.0  # Energia de referência para carbono (α)
        self.alpha_N = 0.5  # α_N = α_C + 0.5β para nitrogênio piridínico
        self.beta_CC = -1.0  # Integral de ressonância C-C
        self.beta_CN = -1.0  # Integral de ressonância C-N
        
    def set_atom_types(self, atom_types):
        """
        Define os tipos de átomos no sistema.
        
        Args:
            atom_types (list): Lista com tipos de átomos ('C' ou 'N')
        """
        if len(atom_types) != self.n_atoms:
            raise ValueError(f"Lista deve ter {self.n_atoms} elementos")
        self.atom_types = atom_types
        
    def set_connectivity(self, bonds):
        """
        Define a conectividade entre átomos.
        
        Args:
            bonds (list): Lista de tuplas (i, j) representando ligações
        """
        self.connectivity = bonds
        
    def build_hamiltonian(self):
        """
        Constrói a matriz Hamiltoniana de Hückel.
        """
        # Zerar a matriz
        self.hamiltonian = np.zeros((self.n_atoms, self.n_atoms))
        
        # Elementos diagonal (energias atômicas)
        for i in range(self.n_atoms):
            if self.atom_types[i] == 'C':
                self.hamiltonian[i, i] = self.alpha_C
            elif self.atom_types[i] == 'N':
                self.hamiltonian[i, i] = self.alpha_N
                
        # Elementos fora da diagonal (integrais de ressonância)
        for i, j in self.connectivity:
            if self.atom_types[i] == 'C' and self.atom_types[j] == 'C':
                beta = self.beta_CC
            else:  # Pelo menos um é N
                beta = self.beta_CN
            self.hamiltonian[i, j] = beta
            self.hamiltonian[j, i] = beta
            
    def solve(self):
        """
        Resolve o problema de autovalores da matriz Hamiltoniana.
        """
        self.eigenvalues, self.eigenvectors = eigh(self.hamiltonian)
        
        # Ordenar por energia crescente
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
    def get_electron_configuration(self, n_electrons):
        """
        Determina a configuração eletrônica.
        
        Args:
            n_electrons (int): Número total de elétrons π
            
        Returns:
            list: Ocupação de cada orbital molecular
        """
        occupations = np.zeros(self.n_atoms)
        electrons_left = n_electrons
        
        for i in range(self.n_atoms):
            if electrons_left >= 2:
                occupations[i] = 2
                electrons_left -= 2
            elif electrons_left == 1:
                occupations[i] = 1
                electrons_left -= 1
            else:
                break
                
        return occupations
        
    def get_homo_lumo(self, n_electrons):
        """
        Identifica os orbitais HOMO e LUMO.
        
        Args:
            n_electrons (int): Número total de elétrons π
            
        Returns:
            tuple: (índice_HOMO, índice_LUMO, energia_HOMO, energia_LUMO)
        """
        occupations = self.get_electron_configuration(n_electrons)
        
        # Encontrar HOMO (último orbital ocupado)
        homo_idx = -1
        for i in range(self.n_atoms):
            if occupations[i] > 0:
                homo_idx = i
                
        # LUMO é o próximo orbital
        lumo_idx = homo_idx + 1 if homo_idx + 1 < self.n_atoms else None
        
        homo_energy = self.eigenvalues[homo_idx] if homo_idx >= 0 else None
        lumo_energy = self.eigenvalues[lumo_idx] if lumo_idx is not None else None
        
        return homo_idx, lumo_idx, homo_energy, lumo_energy
        
    def calculate_bond_orders(self, n_electrons):
        """
        Calcula as ordens de ligação.
        
        Args:
            n_electrons (int): Número total de elétrons π
            
        Returns:
            dict: Ordens de ligação para cada par de átomos conectados
        """
        occupations = self.get_electron_configuration(n_electrons)
        bond_orders = {}
        
        for i, j in self.connectivity:
            order = 0.0
            for k in range(self.n_atoms):
                if occupations[k] > 0:
                    order += occupations[k] * self.eigenvectors[i, k] * self.eigenvectors[j, k]
            bond_orders[(i, j)] = order
            
        return bond_orders
        
    def calculate_electron_populations(self, n_electrons):
        """
        Calcula as populações eletrônicas por átomo.
        
        Args:
            n_electrons (int): Número total de elétrons π
            
        Returns:
            np.array: População eletrônica em cada átomo
        """
        occupations = self.get_electron_configuration(n_electrons)
        populations = np.zeros(self.n_atoms)
        
        for i in range(self.n_atoms):
            for k in range(self.n_atoms):
                if occupations[k] > 0:
                    populations[i] += occupations[k] * self.eigenvectors[i, k]**2
                    
        return populations
        
    def plot_energy_levels(self, n_electrons, title="Diagrama de Níveis de Energia"):
        """
        Plota o diagrama de níveis de energia.
        
        Args:
            n_electrons (int): Número total de elétrons π
            title (str): Título do gráfico
        """
        occupations = self.get_electron_configuration(n_electrons)
        homo_idx, lumo_idx, _, _ = self.get_homo_lumo(n_electrons)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, energy in enumerate(self.eigenvalues):
            # Cor baseada na ocupação
            if occupations[i] == 2:
                color = 'blue'
                label = 'Ocupado (2e⁻)' if i == 0 else ""
            elif occupations[i] == 1:
                color = 'orange'
                label = 'Semi-ocupado (1e⁻)' if i == 0 else ""
            else:
                color = 'red'
                label = 'Vazio' if i == 0 else ""
                
            # Marcar HOMO e LUMO
            if i == homo_idx:
                ax.axhline(y=energy, color=color, linewidth=3, label=f'HOMO {label}')
            elif i == lumo_idx:
                ax.axhline(y=energy, color=color, linewidth=3, label=f'LUMO {label}')
            else:
                ax.axhline(y=energy, color=color, linewidth=2, label=label)
                
            # Adicionar texto com energia
            ax.text(0.1, energy, f'E{i+1} = {energy:.3f}β', 
                   verticalalignment='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylabel('Energia (em unidades de β)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Remover eixo x
        ax.set_xticks([])
        
        plt.tight_layout()
        return fig
        
    def plot_molecular_orbital(self, orbital_idx, atom_positions, title=None):
        """
        Plota um orbital molecular específico.
        
        Args:
            orbital_idx (int): Índice do orbital molecular
            atom_positions (list): Lista de posições (x, y) dos átomos
            title (str): Título do gráfico
        """
        if title is None:
            title = f"Orbital Molecular {orbital_idx + 1}"
            
        coefficients = self.eigenvectors[:, orbital_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plotar átomos
        for i, (x, y) in enumerate(atom_positions):
            coeff = coefficients[i]
            size = abs(coeff) * 1000  # Tamanho proporcional ao coeficiente
            color = 'red' if coeff > 0 else 'blue'
            alpha = min(abs(coeff) * 2, 1.0)  # Transparência baseada no coeficiente
            
            ax.scatter(x, y, s=size, c=color, alpha=alpha, edgecolors='black')
            ax.text(x, y-0.3, f'{i+1}\n{coeff:.3f}', ha='center', va='top', fontsize=8)
            
        # Plotar ligações
        for i, j in self.connectivity:
            x1, y1 = atom_positions[i]
            x2, y2 = atom_positions[j]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, linewidth=1)
            
        ax.set_title(f"{title} (E = {self.eigenvalues[orbital_idx]:.3f}β)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Legenda
        ax.scatter([], [], c='red', s=100, alpha=0.7, label='Fase positiva')
        ax.scatter([], [], c='blue', s=100, alpha=0.7, label='Fase negativa')
        ax.legend()
        
        plt.tight_layout()
        return fig

