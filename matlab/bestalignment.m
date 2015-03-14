function [score, alignment ] = bestalignment(trip1, trip2)
    matrix = 'NUC44';
    gap = 10000;
    
    [score, alignment]   = nwalign(trip1, trip2, 'Alphabet', 'NT', 'ScoringMatrix', matrix, 'GapOpen', gap, 'EXTENDGAP', 0);
   
    [score2, alignment2] = nwalign(trip1, fliplr(trip2), 'Alphabet', 'NT', 'ScoringMatrix', matrix, 'GapOpen', gap, 'EXTENDGAP', 0);
    if score2 > score
        score = score2;
        alignment = alignment2;
    end
    trip2mirrored = strrep(trip2, 'C', 'T');
    trip2mirrored = strrep(trip2mirrored, 'A', 'C');
    trip2mirrored = strrep(trip2mirrored, 'T', 'A');
    
    [score2, alignment2] = nwalign(trip1, trip2mirrored, 'Alphabet', 'NT', 'ScoringMatrix', matrix, 'GapOpen', gap, 'EXTENDGAP', 0);
    if score2 > score
        score = score2;
        alignment = alignment2;
    end
    
    [score2, alignment2] = nwalign(trip1, fliplr(trip2mirrored), 'Alphabet', 'NT', 'ScoringMatrix', matrix, 'GapOpen', gap, 'EXTENDGAP', 0);
    if score2 > score
        score = score2;
        alignment = alignment2;
    end 
end

