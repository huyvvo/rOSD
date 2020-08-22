classdef DataLoader
  properties
    S   % matrix of cells of similarity
    proposal_indices % 
  end

  methods
    function self = DataLoader(scores)
      self.S = scores.S;
      if ~isfield(scores, 'proposal_indices')
        self.proposal_indices = [];
      else
        self.proposal_indices = scores.proposal_indices;
      end
    end % function

    function res = get_S(self, i,j)
      if i < j
        res = self.S{i,j};
      elseif i > j
        res = transpose(self.S{j,i});
      else
        error('i and j must be different...');
      end
      if ~isempty(self.proposal_indices)
        res = res(self.proposal_indices{i}, self.proposal_indices{j});
      end
    end % function

    function obj = set.S(obj, S)
      obj.S = S;
    end % function

    function obj = set.proposal_indices(obj, proposal_indices)
      obj.proposal_indices = proposal_indices;
    end % function

  end % method

end % class