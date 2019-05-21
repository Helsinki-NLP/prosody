#!/usr/bin/env perl

my $goldfile = shift(@ARGV) || die "no gold standard file given!\n";
open G,"<$goldfile" || die "cannot read from $goldfile!\n";


while (<>){
    chomp;
    next unless (/\S/);
    my ($label) = split(/\s/);
    my $gold = <G>;
    unless ($gold=~/\S/){$gold = <G>};
    chomp($gold);
    my ($gold) = split(/\s/,$gold);

    $totalAll++;
    $totalGold{$gold}++;
    $totalSys{$label}++;

    if ($gold eq $label){
	$correctAll++;
	$correct{$gold}++;
    }
}

printf "\taccuracy\t%5.3f (%d/%d)\n\n",100*$correctAll/$totalAll,$correctAll,$totalAll;

foreach my $l (sort keys %totalSys){
    printf "($l)\tprecision\t%5.3f (%d/%d)\n",100*$correct{$l}/$totalSys{$l},$correct{$l},$totalSys{$l};
    printf "    \trecall   \t%5.3f (%d/%d)\n",100*$correct{$l}/$totalGold{$l},$correct{$l},$totalGold{$l};
}
