import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import networkx as nx

class HuckelMethodParametros:
    """
    Classe para implementar o método de Hückel com parâmetros específicos
    baseados na tabela fornecida.
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
        
        # Parâmetros da tabela fornecida
        # Parâmetros h (energias atômicas em unidades de β)
        self.h_params = {
            'B': -1.0,
            'C': 0.0,
            'N': 0.5,  # Usando hN para nitrogênio piridínico
            'O': 1.0,
            'F': 3.0,
            'Cl': 2.0,
            'Br': 1.5
        }
        
        # Parâmetros k (integrais de ressonância como fração de β)
        self.k_params = {
            ('B', 'C'): 0.7,
            ('C', 'B'): 0.7,
            ('C', 'C'): 1.0,  # Usando kC-C = 1.0
            ('C', 'N'): 1.0,  # Usando kCN = 1.0
            ('N', 'C'): 1.0,
            ('N', 'N'): 0.8,
            ('C', 'O'): 1.0,
            ('O', 'C'): 1.0,
            ('C', 'F'): 0.7,
            ('F', 'C'): 0.7,
            ('C', 'Cl'): 0.4,
            ('Cl', 'C'): 0.4,
            ('C', 'Br'): 0.3,
            ('Br', 'C'): 0.3
        }
        
    def set_atom_types(self, atom_types):
        """
        Define os tipos de átomos no sistema.
        
        Args:
            atom_types (list): Lista com tipos de átomos ('C', 'N', 'O', etc.)
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
        Constrói a matriz Hamiltoniana de Hückel usando os parâmetros da tabela.
        """
        # Zerar a matriz
        self.hamiltonian = np.zeros((self.n_atoms, self.n_atoms))
        
        # Elementos diagonal (energias atômicas)
        for i in range(self.n_atoms):
            atom_type = self.atom_types[i]
            if atom_type in self.h_params:
                self.hamiltonian[i, i] = self.h_params[atom_type]
            else:
                # Valor padrão para carbono
                self.hamiltonian[i, i] = 0.0
                
        # Elementos fora da diagonal (integrais de ressonância)
        for i, j in self.connectivity:
            atom_i = self.atom_types[i]
            atom_j = self.atom_types[j]
            
            # Buscar parâmetro k para o par de átomos
            if (atom_i, atom_j) in self.k_params:
                k_value = self.k_params[(atom_i, atom_j)]
            elif (atom_j, atom_i) in self.k_params:
                k_value = self.k_params[(atom_j, atom_i)]
            else:
                # Valor padrão
                k_value = 1.0
                
            # β é negativo por convenção, k é o fator multiplicativo
            beta_ij = -k_value
            self.hamiltonian[i, j] = beta_ij
            self.hamiltonian[j, i] = beta_ij
            
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
            ax.text(x, y-0.3, f'{i+1}\\n{self.atom_types[i]}\\n{coeff:.3f}', 
                   ha='center', va='top', fontsize=8)
            
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
        
    def print_parameters_used(self):
        """
        Imprime os parâmetros utilizados nos cálculos.
        """
        print("PARÂMETROS UTILIZADOS (baseados na tabela fornecida):")
        print("=" * 60)
        
        print("\\nParâmetros h (energias atômicas em unidades de β):")
        for atom, h_val in self.h_params.items():
            print(f"  h_{atom} = {h_val}")
            
        print("\\nParâmetros k (integrais de ressonância como fração de β):")
        for bond, k_val in self.k_params.items():
            print(f"  k_{bond[0]}-{bond[1]} = {k_val}")
            
        print("\\nNota: β < 0 por convenção, então β_ij = -k_ij × β")

