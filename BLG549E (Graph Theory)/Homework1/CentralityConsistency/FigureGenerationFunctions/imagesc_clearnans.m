function h = imagesc_clearnans(data)
% a wrapper for imagesc, with some formatting going on for nans

% plotting data. Removing and scaling axes (this is for image plotting)
h = imagesc(data);
%axis image off

% setting alpha values
if ismatrix(data)
  set(h, 'AlphaData', ~isnan(data))
elseif ndims(data) == 3
  set(h, 'AlphaData', ~isnan(data(:, :, 1)))
end

if nargout < 1
  clear h
end

end